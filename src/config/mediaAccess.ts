import { mkdirSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import process from 'node:process';

/** Environment variable holding the path-delimited list of directories media tools may read. */
export const MEDIA_DIRS_ENV = 'EBAY_MCP_MEDIA_DIRS';

/** Environment variable holding the single root that `media://` references resolve under. */
export const MEDIA_ROOT_ENV = 'EBAY_MCP_MEDIA_ROOT';

/** Parsed local-media allowlist. */
export interface MediaAccessConfig {
  /** Absolute directories the media tools may read from (the root is included when set). */
  readonly allowedDirs: readonly string[];
  /** Absolute directory that `media://<relative>` references resolve under. */
  readonly mediaRoot?: string;
  /** Configuration problems, reported by startup validation and by the tools themselves. */
  readonly errors: readonly string[];
}

const parseAbsoluteDir = (raw: string, variable: string, errors: string[]): string | undefined => {
  if (!path.isAbsolute(raw)) {
    errors.push(`${variable} entries must be absolute paths; got "${raw}".`);
    return;
  }
  return path.resolve(raw);
};

/**
 * Reads the local-media allowlist from the environment.
 *
 * Local file access is off unless the operator names the directories the server
 * may read: `EBAY_MCP_MEDIA_DIRS` (a `path.delimiter`-separated list) and/or
 * `EBAY_MCP_MEDIA_ROOT` (one directory that also anchors `media://` references).
 *
 * @param env - Environment to read; defaults to `process.env`.
 * @returns The allowed directories, the optional media root, and any configuration errors.
 *
 * @example
 * ```ts
 * const access = getMediaAccessConfig();
 * if (access.allowedDirs.length === 0) {
 *   // media tools stay disabled
 * }
 * ```
 */
export const getMediaAccessConfig = (env: NodeJS.ProcessEnv = process.env): MediaAccessConfig => {
  const errors: string[] = [];
  const allowedDirs: string[] = [];

  const rawDirs = (env[MEDIA_DIRS_ENV] ?? '')
    .split(path.delimiter)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
  for (const raw of rawDirs) {
    const dir = parseAbsoluteDir(raw, MEDIA_DIRS_ENV, errors);
    if (dir && !allowedDirs.includes(dir)) {
      allowedDirs.push(dir);
    }
  }

  const rawRoot = (env[MEDIA_ROOT_ENV] ?? '').trim();
  const mediaRoot = rawRoot ? parseAbsoluteDir(rawRoot, MEDIA_ROOT_ENV, errors) : undefined;
  if (mediaRoot && !allowedDirs.includes(mediaRoot)) {
    allowedDirs.push(mediaRoot);
  }

  return { allowedDirs, mediaRoot, errors };
};

/** Default photo folder suggested by the setup wizard (`~/Claude/ebay-fotos`). */
export const getDefaultMediaDir = (): string => path.join(homedir(), 'Claude', 'ebay-fotos');

/** Result of {@link ensureMediaDirs}. */
export interface EnsureMediaDirsResult {
  /** Directories that did not exist and were created. */
  readonly created: string[];
  /** Directories that could not be created, with the reason. */
  readonly failed: { dir: string; message: string }[];
}

/**
 * Creates every configured media directory that does not exist yet, private to
 * the current user (mode 0700), so a fresh setup works without a manual
 * `mkdir`. Existing directories are left exactly as they are. Nothing happens
 * when no media directory is configured, which keeps local file access opt-in.
 *
 * @param access - Parsed media allowlist; defaults to the current environment.
 * @returns The directories created and any that failed.
 */
export const ensureMediaDirs = (
  access: MediaAccessConfig = getMediaAccessConfig(),
): EnsureMediaDirsResult => {
  const created: string[] = [];
  const failed: { dir: string; message: string }[] = [];
  for (const dir of access.allowedDirs) {
    try {
      statSync(dir);
    } catch {
      try {
        mkdirSync(dir, { recursive: true, mode: 0o700 });
        created.push(dir);
      } catch (error) {
        failed.push({ dir, message: error instanceof Error ? error.message : String(error) });
      }
    }
  }
  return { created, failed };
};
