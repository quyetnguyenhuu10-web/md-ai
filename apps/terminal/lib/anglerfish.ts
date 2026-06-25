#!/usr/bin/env node

/**
 * Cá đèn pixel chuyển động nhẹ trong terminal.
 *
 * - Cá đứng nguyên vị trí.
 * - Đầu, thân, mắt và cần đèn không dao động.
 * - Chỉ phần ngoài của đuôi quẫy nhẹ.
 * - Nước trôi chậm từ phải sang trái.
 * - Phần cá dưới nước có màu tối hơn.
 *
 * Chạy:
 *   bun apps/terminal/lib/anglerfish.ts
 *
 * Tắt màu:
 *   bun apps/terminal/lib/anglerfish.ts --no-color
 *
 * Điều chỉnh tốc độ:
 *   bun apps/terminal/lib/anglerfish.ts --fps 10
 */

const RESET = "\x1b[0m";
const HOME = "\x1b[H";
export const CLEAR = "\x1b[2J";
const HIDE_CURSOR = "\x1b[?25l";
const SHOW_CURSOR = "\x1b[?25h";

const BODY = "\x1b[38;2;92;220;180m";
const TAIL = "\x1b[38;2;34;170;150m";

const BODY_SHADOW = "\x1b[2;38;2;62;150;132m";
const TAIL_SHADOW = "\x1b[2;38;2;38;128;114m";

const EYE = "\x1b[38;2;235;255;250m";
const STALK = "\x1b[38;2;150;245;220m";
const LURE = "\x1b[38;2;255;210;70m";

const WATER = "\x1b[38;2;105;190;215m";
const WATER_LIGHT = "\x1b[38;2;170;225;235m";
const WATER_DIM = "\x1b[2;38;2;80;145;170m";

const WATER_START_ROW = 6;
const TAIL_END_COLUMN = 9;
const CANVAS_WIDTH = 42;
const BOTTOM_WATER_ROWS = 4;

type Options = {
    fps: number;
    useColor: boolean;
};

/**
 * Frame 0 là dáng trung tâm.
 * Frame 1 và 2 chỉ thay đổi phần ngoài cùng của đuôi.
 *
 * Đầu, mắt, miệng, thân và cần đèn giữ nguyên.
 */
const FISH_FRAMES: readonly (readonly string[])[] = [
    [
        "                    ██",
        "                   █ █",
        "                  ██",
        "             ██████",
        "    ██    █████████████",
        "  █████████████████░████",
        "█████████████████████████",
        "  █████████████████████",
        "    ██    ████████████████",
        "             ██████████",
        "                ████",
    ],
    [
        "                    ██",
        "                   █ █",
        "                  ██",
        "             ██████",
        "   ██     █████████████",
        "  █████████████████░████",
        " ████████████████████████",
        "  █████████████████████",
        "     ██   ████████████████",
        "             ██████████",
        "                ████",
    ],
    [
        "                    ██",
        "                   █ █",
        "                  ██",
        "             ██████",
        "     ██   █████████████",
        "  █████████████████░████",
        " ████████████████████████",
        "  █████████████████████",
        "   ██     ████████████████",
        "             ██████████",
        "                ████",
    ],
];

/**
 * Giữ frame trung tâm lâu hơn.
 *
 * Trình tự:
 * giữa → trên → giữa → dưới → giữa
 */
const TAIL_SEQUENCE = [
    0, 0, 0, 0,
    1, 1,
    0, 0, 0, 0,
    2, 2,
] as const;

/**
 * Hoa văn nước thưa để không gây rối mắt.
 */
const WATER_PATTERNS = [
    "≈          ≈              ≈         ≈             ",
    "      ≈              ≈          ≈              ≈  ",
    "  ≈             ≈          ≈              ≈       ",
    "         ≈           ≈              ≈          ≈   ",
] as const;

function parseArguments(argumentsList: readonly string[]): Options {
    let fps = 12;
    let useColor = !argumentsList.includes("--no-color");

    const fpsIndex = argumentsList.indexOf("--fps");

    if (fpsIndex !== -1) {
        const value = argumentsList[fpsIndex + 1];

        if (value === undefined) {
            throw new Error("--fps cần một giá trị.");
        }

        fps = Number(value);
    }

    if (!Number.isFinite(fps) || fps <= 0) {
        throw new Error("--fps phải là số lớn hơn 0.");
    }

    if (process.env.NO_COLOR !== undefined) {
        useColor = false;
    }

    return {
        fps,
        useColor,
    };
}

function color(
    text: string,
    ansi: string,
    enabled: boolean,
): string {
    if (!enabled) {
        return text;
    }

    return `${ansi}${text}${RESET}`;
}

function getFishColor(
    row: number,
    column: number,
    character: string,
): string {
    if (character === "░") {
        return EYE;
    }

    if (row === 0 || row === 1) {
        return LURE;
    }

    if (row === 2 || row === 3) {
        return STALK;
    }

    const isTail =
        row >= 4
        && column < TAIL_END_COLUMN;

    const isUnderwater =
        row > WATER_START_ROW;

    if (isTail && isUnderwater) {
        return TAIL_SHADOW;
    }

    if (isUnderwater) {
        return BODY_SHADOW;
    }

    if (isTail) {
        return TAIL;
    }

    return BODY;
}

function getWaterPixel(
    row: number,
    column: number,
    phase: number,
): string {
    const pattern =
        WATER_PATTERNS[row % WATER_PATTERNS.length];

    const sourceColumn =
        (column + phase) % pattern.length;

    return pattern[sourceColumn] ?? " ";
}

export function renderAnglerfishBanner(
    options: {
        useColor?: boolean;
        waterPhase?: number;
        frameIndex?: number;
    } = {},
): string {
    const useColor =
        options.useColor
        ?? process.env.NO_COLOR === undefined;
    const fish =
        FISH_FRAMES[options.frameIndex ?? 0]
        ?? FISH_FRAMES[0];

    if (fish === undefined) {
        throw new Error(
            "Không tìm thấy frame cá.",
        );
    }

    return renderFrame(
        fish,
        options.waterPhase ?? 0,
        useColor,
    );
}

export function renderAnglerfishFrame(
    options: {
        tick: number;
        useColor?: boolean;
    },
): string {
    const sequenceIndex =
        Math.floor(options.tick / 3)
        % TAIL_SEQUENCE.length;

    const fishFrameIndex =
        TAIL_SEQUENCE[sequenceIndex] ?? 0;

    return renderAnglerfishBanner({
        frameIndex: fishFrameIndex,
        waterPhase: Math.floor(options.tick / 3),
        useColor: options.useColor,
    });
}

function renderFrame(
    fish: readonly string[],
    waterPhase: number,
    useColor: boolean,
): string {
    const lines: string[] = [];

    for (
        let rowIndex = 0;
        rowIndex < fish.length;
        rowIndex += 1
    ) {
        const fishRow =
            (fish[rowIndex] ?? "").padEnd(CANVAS_WIDTH);

        const renderedRow: string[] = [];

        for (
            let column = 0;
            column < CANVAS_WIDTH;
            column += 1
        ) {
            const character =
                fishRow[column] ?? " ";

            if (character !== " ") {
                renderedRow.push(
                    color(
                        character,
                        getFishColor(
                            rowIndex,
                            column,
                            character,
                        ),
                        useColor,
                    ),
                );

                continue;
            }

            if (rowIndex < WATER_START_ROW) {
                renderedRow.push(" ");
                continue;
            }

            const wave = getWaterPixel(
                rowIndex,
                column,
                waterPhase,
            );

            if (wave === " ") {
                renderedRow.push(" ");
                continue;
            }

            const waveColor =
                rowIndex === WATER_START_ROW
                    ? WATER_LIGHT
                    : WATER_DIM;

            renderedRow.push(
                color(
                    wave,
                    waveColor,
                    useColor,
                ),
            );
        }

        lines.push(renderedRow.join(""));
    }

    for (
        let waterRow = 0;
        waterRow < BOTTOM_WATER_ROWS;
        waterRow += 1
    ) {
        const renderedRow: string[] = [];

        for (
            let column = 0;
            column < CANVAS_WIDTH;
            column += 1
        ) {
            const wave = getWaterPixel(
                fish.length + waterRow,
                column,
                waterPhase,
            );

            if (wave === " ") {
                renderedRow.push(" ");
                continue;
            }

            const waveColor =
                waterRow === 0
                    ? WATER
                    : WATER_DIM;

            renderedRow.push(
                color(
                    wave,
                    waveColor,
                    useColor,
                ),
            );
        }

        lines.push(renderedRow.join(""));
    }

    return lines.join("\n");
}

function sleep(milliseconds: number): Promise<void> {
    return new Promise((resolve) => {
        setTimeout(resolve, milliseconds);
    });
}

async function animate(options: Options): Promise<void> {
    const frameDuration = 1000 / options.fps;

    let tick = 0;
    let running = true;

    const stop = (): void => {
        running = false;
    };

    process.once("SIGINT", stop);
    process.once("SIGTERM", stop);

    process.stdout.write(
        CLEAR
        + HOME
        + HIDE_CURSOR,
    );

    try {
        while (running) {
            const startedAt =
                performance.now();

            /**
             * Mỗi trạng thái đuôi được giữ qua ba lần vẽ.
             * Điều này làm chuyển động chậm và ít rung.
             */
            const frame = renderAnglerfishFrame({
                tick,
                useColor: options.useColor,
            });

            process.stdout.write(HOME);
            process.stdout.write(frame);
            process.stdout.write(RESET);

            tick += 1;

            const elapsed =
                performance.now() - startedAt;

            const remaining =
                frameDuration - elapsed;

            if (remaining > 0) {
                await sleep(remaining);
            }
        }
    } finally {
        process.stdout.write(
            RESET
            + SHOW_CURSOR
            + "\n",
        );
    }
}

async function main(): Promise<void> {
    try {
        const options = parseArguments(
            process.argv.slice(2),
        );

        await animate(options);
    } catch (error: unknown) {
        process.stdout.write(
            RESET
            + SHOW_CURSOR,
        );

        const message =
            error instanceof Error
                ? error.message
                : String(error);

        console.error(`Lỗi: ${message}`);
        process.exitCode = 1;
    }
}

function isDirectRun(): boolean {
    const entrypoint =
        process.argv[1];

    if (entrypoint === undefined) {
        return false;
    }

    return import.meta.url.endsWith(
        entrypoint.replaceAll("\\", "/"),
    );
}

if (isDirectRun()) {
    void main();
}
