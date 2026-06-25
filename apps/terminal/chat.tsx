import { render } from "@opentui/solid";
import { App } from "./App";
import { renderAnglerfishBanner } from "./lib/anglerfish";

const CLEAR_VISIBLE_SCREEN = "\x1b[2J";
const HOME = "\x1b[H";

function drawStartupScreen() {
  if (!process.stdout.isTTY) return;
  process.stdout.write(CLEAR_VISIBLE_SCREEN + HOME);
  process.stdout.write(renderAnglerfishBanner());
  process.stdout.write("\n");
}

drawStartupScreen();

render(() => <App />, {
  screenMode: "split-footer",
  externalOutputMode: "capture-stdout",
  footerHeight: 3,
  clearOnShutdown: false,
});
