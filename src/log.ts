/** The page log (public/log.js, loaded before the game): what happened, shown with Copy and Share. */
interface PageLog {
  info(text: string, detail?: string): void;
  warn(text: string, detail?: string): void;
  error(text: string, detail?: string): void;
  /** A thrown value as [one-line text, detail (its stack)]. */
  describe(value: unknown): [string, string];
  /** Show the log panel. */
  open(): void;
}

/** Without the page log (Node tests), the console. */
const consoleLog: PageLog = {
  info: () => {},
  warn: (text, detail) => console.warn(text, detail ?? ''),
  error: (text, detail) => console.error(text, detail ?? ''),
  describe: (value) => (value instanceof Error ? [`${value.name}: ${value.message}`, value.stack ?? ''] : [String(value), '']),
  open: () => {},
};

const page = (): PageLog => (globalThis as { blockLog?: PageLog }).blockLog ?? consoleLog;

export const log: PageLog = {
  info: (text, detail) => page().info(text, detail),
  warn: (text, detail) => page().warn(text, detail),
  error: (text, detail) => page().error(text, detail),
  describe: (value) => page().describe(value),
  open: () => page().open(),
};

/** Log a thrown value as an error, with what was being done. */
export function logError(what: string, value: unknown): void {
  const [text, stack] = log.describe(value);
  log.error(`${what}: ${text}`, stack);
}
