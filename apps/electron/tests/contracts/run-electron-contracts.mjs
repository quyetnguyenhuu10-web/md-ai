import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import * as yaml from "js-yaml";
import { parse } from "@typescript-eslint/typescript-estree";

const SOURCE_SUFFIXES = [".ts", ".tsx", ".js", ".jsx"];
const KNOWN_RULES = new Set([
  "forbid_import_cycles",
  "forbid_private_cross_imports",
  "forbid_relative_parent_imports",
  "forbid_top_level_io",
  "forbid_runtime_wiring_outside_composition_root",
  "require_layer_markers",
  "strict_public_api",
  "allow_private_imports_for_tests",
]);

const repoRoot = findElectronRoot(process.cwd());
const config = loadYaml(path.join(repoRoot, "tests", "contracts", "electron_contract.yaml"));
validateConfig(config);
const baseline = loadBaseline(config);
const tsFiles = discoverSourceFiles();
const imports = tsFiles.flatMap((file) => collectImports(file));

const violations = [
  ...layerImportBoundaryViolations(),
  ...privateImportViolations(),
  ...relativeParentImportViolations(),
  ...runtimeWiringViolations(),
  ...cycleViolations(),
  ...publicApiViolations(),
].filter((violation) => !baseline.has(violationKey(violation)));

if (violations.length > 0) {
  console.error(formatViolations(violations));
  process.exit(1);
}

console.log(`architecture contract TS scan passed (${tsFiles.length} files, ${imports.length} imports)`);

function findElectronRoot(start) {
  let current = path.resolve(start);
  while (true) {
    if (
      fs.existsSync(path.join(current, "package.json")) &&
      fs.existsSync(path.join(current, "tests", "contracts", "electron_contract.yaml"))
    ) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      throw new Error("Could not find apps/electron root with tests/contracts/electron_contract.yaml");
    }
    current = parent;
  }
}

function loadYaml(filePath) {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  return yaml.load(fs.readFileSync(filePath, "utf8")) ?? {};
}

function loadBaseline(contract) {
  const file = contract.baseline?.file;
  if (!file) {
    return new Set();
  }
  const data = loadYaml(path.join(repoRoot, file));
  return new Set((data.violations ?? []).map(String));
}

function validateConfig(contract) {
  const rules = contract.rules ?? {};
  const unknownRules = Object.keys(rules).filter((rule) => !KNOWN_RULES.has(rule));
  if (unknownRules.length > 0) {
    throw new Error(`architecture_contract.yaml unknown rules: ${unknownRules.join(", ")}`);
  }
  for (const root of contract.source_roots ?? []) {
    if (!root || typeof root !== "string") {
      throw new Error("architecture_contract.yaml source_roots must contain non-empty strings");
    }
  }
  for (const [name, layer] of Object.entries(contract.layers ?? {})) {
    if (!name.trim()) {
      throw new Error("architecture_contract.yaml layer names must be non-empty");
    }
    if (!Array.isArray(layer.patterns) || layer.patterns.some((pattern) => !pattern)) {
      throw new Error(`architecture_contract.yaml layer ${name} must define non-empty patterns`);
    }
  }
}

function discoverSourceFiles() {
  const files = [];
  for (const sourceRoot of config.source_roots ?? []) {
    const absoluteRoot = path.join(repoRoot, sourceRoot);
    if (!fs.existsSync(absoluteRoot)) {
      continue;
    }
    for (const file of walk(absoluteRoot)) {
      if (SOURCE_SUFFIXES.includes(path.extname(file)) && !isIgnored(file)) {
        files.push(file);
      }
    }
  }
  return [...new Set(files)].sort();
}

function* walk(root) {
  const stat = fs.statSync(root);
  if (stat.isFile()) {
    yield root;
    return;
  }
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const absolute = path.join(root, entry.name);
    if (isIgnored(absolute)) {
      continue;
    }
    if (entry.isDirectory()) {
      yield* walk(absolute);
    } else {
      yield absolute;
    }
  }
}

function isIgnored(file) {
  const relative = rel(file);
  const parts = relative.split("/");
  return (config.ignored_dirs ?? []).some((ignored) => {
    if (!ignored.includes("/") && parts.includes(ignored)) {
      return true;
    }
    return matchesGlob(relative, ignored) || matchesGlob(relative, `${ignored}/**`) || matchesGlob(relative, `**/${ignored}/**`);
  });
}

function collectImports(file) {
  const text = fs.readFileSync(file, "utf8");
  const ast = parse(text, {
    loc: true,
    jsx: file.endsWith(".tsx") || file.endsWith(".jsx"),
    sourceType: "module",
    comment: false,
    range: false,
  });
  const records = [];

  for (const node of ast.body) {
    if (node.type === "ImportDeclaration") {
      const source = String(node.source.value);
      const resolved = resolveImport(file, source);
      for (const specifier of node.specifiers) {
        records.push({
          file,
          line: node.loc?.start.line ?? 1,
          source,
          importedName: importedSpecifierName(specifier),
          resolved,
          kind: "import",
        });
      }
      if (node.specifiers.length === 0) {
        records.push({ file, line: node.loc?.start.line ?? 1, source, importedName: null, resolved, kind: "import" });
      }
    }
    if ((node.type === "ExportNamedDeclaration" || node.type === "ExportAllDeclaration") && node.source) {
      const source = String(node.source.value);
      records.push({
        file,
        line: node.loc?.start.line ?? 1,
        source,
        importedName: null,
        resolved: resolveImport(file, source),
        kind: "export",
      });
    }
  }

  return records;
}

function importedSpecifierName(specifier) {
  if (specifier.type === "ImportSpecifier") {
    const imported = specifier.imported;
    return imported.type === "Identifier" ? imported.name : imported.value;
  }
  if (specifier.type === "ImportDefaultSpecifier") {
    return "default";
  }
  if (specifier.type === "ImportNamespaceSpecifier") {
    return "*";
  }
  return null;
}

function resolveImport(importer, source) {
  if (!source.startsWith(".")) {
    return null;
  }
  const base = path.resolve(path.dirname(importer), source);
  const candidates = [];
  for (const suffix of SOURCE_SUFFIXES) {
    candidates.push(`${base}${suffix}`);
  }
  for (const suffix of SOURCE_SUFFIXES) {
    candidates.push(path.join(base, `index${suffix}`));
  }
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

function layerImportBoundaryViolations() {
  const violations = [];
  for (const record of imports) {
    const sourceLayer = layerForPath(record.file);
    if (!sourceLayer) {
      continue;
    }
    const importedLayer = record.resolved ? layerForPath(record.resolved) : null;
    const layerConfig = config.layers[sourceLayer] ?? {};
    const forbidden = [...(config.forbidden_imports ?? []), ...(layerConfig.forbidden_imports ?? [])];
    for (const pattern of forbidden) {
      if (importMatches(pattern, record.source, importedLayer)) {
        violations.push({
          rule: "ts_layer_forbidden_import",
          path: rel(record.file),
          line: record.line,
          detail: `layer=${sourceLayer} import=${record.source} matched_forbidden=${pattern}`,
        });
      }
    }
    const allowed = [...(config.allowed_imports ?? []), ...(layerConfig.allowed_imports ?? [])];
    if (!isImportAllowed(record, importedLayer, allowed)) {
      violations.push({
        rule: "ts_layer_import_not_allowed",
        path: rel(record.file),
        line: record.line,
        detail: `layer=${sourceLayer} import=${record.source} resolved=${record.resolved ? rel(record.resolved) : "<external>"}`,
      });
    }
  }
  return violations;
}

function isImportAllowed(record, importedLayer, allowed) {
  if (importedLayer && allowed.includes(importedLayer)) {
    return true;
  }
  if (!record.resolved && allowed.includes("external")) {
    return true;
  }
  if (!record.resolved && allowed.includes("stdlib") && (record.source.startsWith("node:") || ["fs", "path", "process"].includes(record.source))) {
    return true;
  }
  return allowed.some((pattern) => importMatches(pattern, record.source, importedLayer));
}

function privateImportViolations() {
  if (!config.rules?.forbid_private_cross_imports) {
    return [];
  }
  return imports
    .filter((record) => record.importedName?.startsWith("_") && !record.importedName.startsWith("__"))
    .map((record) => ({
      rule: "ts_private_cross_import",
      path: rel(record.file),
      line: record.line,
      detail: `module=${record.source} private_name=${record.importedName}`,
    }));
}

function relativeParentImportViolations() {
  if (!config.rules?.forbid_relative_parent_imports) {
    return [];
  }
  return imports
    .filter((record) => record.source.startsWith("../"))
    .filter((record) => {
      const sourceLayer = layerForPath(record.file);
      const importedLayer = record.resolved ? layerForPath(record.resolved) : null;
      if (!sourceLayer || !importedLayer || sourceLayer === importedLayer) {
        return false;
      }
      const layerConfig = config.layers[sourceLayer] ?? {};
      const allowed = [...(config.allowed_imports ?? []), ...(layerConfig.allowed_imports ?? [])];
      return !allowed.includes(importedLayer);
    })
    .map((record) => ({
      rule: "ts_relative_parent_cross_layer_import",
      path: rel(record.file),
      line: record.line,
      detail: `import=${record.source} source_layer=${layerForPath(record.file)} imported_layer=${layerForPath(record.resolved)}`,
    }));
}

function runtimeWiringViolations() {
  if (!config.rules?.forbid_runtime_wiring_outside_composition_root) {
    return [];
  }
  const runtimeImports = config.runtime_wiring_imports ?? [];
  return imports
    .filter((record) => !isCompositionRoot(record.file))
    .flatMap((record) =>
      runtimeImports
        .filter((runtime) => importMatches(runtime, record.source, null))
        .map((runtime) => ({
          rule: "ts_runtime_wiring_import_outside_composition_root",
          path: rel(record.file),
          line: record.line,
          detail: `import=${record.source} matched=${runtime}`,
        })),
    );
}

function cycleViolations() {
  if (!config.rules?.forbid_import_cycles) {
    return [];
  }
  const graph = new Map(tsFiles.map((file) => [file, new Set()]));
  for (const record of imports) {
    if (record.resolved && record.resolved !== record.file) {
      graph.get(record.file)?.add(record.resolved);
    }
  }
  return findCycles(graph).map((cycle) => ({
    rule: "ts_project_import_cycle",
    path: "<dependency-graph>",
    line: 0,
    detail: cycle.map(rel).join(" -> "),
  }));
}

function publicApiViolations() {
  const violations = [];
  for (const contract of config.public_api_contracts ?? []) {
    if (!contract.module?.startsWith("apps/")) {
      continue;
    }
    const target = path.join(repoRoot, contract.module);
    const actual = exportedNamesForFile(target);
    const expected = new Set(contract.expected ?? []);
    const missing = [...expected].filter((name) => !actual.has(name));
    const extra = [...actual].filter((name) => !expected.has(name));
    if (missing.length > 0) {
      violations.push({ rule: "ts_public_api_missing", path: rel(target), line: 0, detail: `missing=${missing.join(",")}` });
    }
    if (contract.mode === "exact" && extra.length > 0) {
      violations.push({ rule: "ts_public_api_extra", path: rel(target), line: 0, detail: `extra=${extra.join(",")}` });
    }
  }
  return violations;
}

function exportedNamesForFile(file) {
  if (!fs.existsSync(file)) {
    return new Set();
  }
  const ast = parse(fs.readFileSync(file, "utf8"), { loc: true, jsx: file.endsWith(".tsx"), sourceType: "module" });
  const names = new Set();
  for (const node of ast.body) {
    if (node.type === "ExportNamedDeclaration") {
      for (const specifier of node.specifiers ?? []) {
        names.add(specifier.exported.type === "Identifier" ? specifier.exported.name : specifier.exported.value);
      }
      if (node.declaration?.id?.name) {
        names.add(node.declaration.id.name);
      }
    }
  }
  return names;
}

function findCycles(graph) {
  const cycles = [];
  const visited = new Set();
  const active = new Set();
  const stack = [];

  function visit(node) {
    visited.add(node);
    active.add(node);
    stack.push(node);
    for (const next of graph.get(node) ?? []) {
      if (!visited.has(next)) {
        visit(next);
      } else if (active.has(next)) {
        const start = stack.indexOf(next);
        cycles.push([...stack.slice(start), next]);
      }
    }
    stack.pop();
    active.delete(node);
  }

  for (const node of graph.keys()) {
    if (!visited.has(node)) {
      visit(node);
    }
  }
  return dedupeCycles(cycles);
}

function dedupeCycles(cycles) {
  const seen = new Set();
  return cycles.filter((cycle) => {
    const key = cycle.map(rel).sort().join("|");
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function layerForPath(file) {
  if (!file) {
    return null;
  }
  const relative = rel(file);
  for (const [name, layer] of Object.entries(config.layers ?? {})) {
    if ((layer.patterns ?? []).some((pattern) => matchesGlob(relative, pattern))) {
      return name;
    }
  }
  return null;
}

function isCompositionRoot(file) {
  const relative = rel(file);
  return (config.composition_roots ?? []).some((pattern) => matchesGlob(relative, pattern));
}

function importMatches(pattern, source, importedLayer) {
  if (importedLayer && pattern === importedLayer) {
    return true;
  }
  return matchesGlob(source, pattern) || source === pattern || source.startsWith(`${pattern}/`) || source.startsWith(`${pattern}.`);
}

function matchesGlob(value, pattern) {
  const regex = new RegExp(`^${globToRegex(pattern)}$`);
  return regex.test(value);
}

function globToRegex(pattern) {
  return pattern
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, "<<<GLOBSTAR>>>")
    .replace(/\*/g, "[^/]*")
    .replace(/<<<GLOBSTAR>>>/g, ".*");
}

function rel(file) {
  return path.relative(repoRoot, file).replaceAll(path.sep, "/");
}

function violationKey(violation) {
  return `${violation.rule}|${violation.path}|${violation.line}|${violation.detail}`;
}

function formatViolations(items) {
  const rendered = items
    .slice(0, 120)
    .map((violation) => `${violation.path}:${violation.line}: ${violation.rule}: ${violation.detail}`)
    .join("\n");
  const extra = items.length > 120 ? `\n... ${items.length - 120} more` : "";
  return `${rendered}${extra}`;
}
