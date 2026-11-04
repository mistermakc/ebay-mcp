import { mkdir, mkdtemp, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { ensureMediaDirs, getMediaAccessConfig } from '@/config/mediaAccess.js';
import { describe, expect, it } from 'vitest';

describe('getMediaAccessConfig', () => {
  it('disables local media access when nothing is configured', () => {
    const access = getMediaAccessConfig({});

    expect(access.allowedDirs).toEqual([]);
    expect(access.mediaRoot).toBeUndefined();
    expect(access.errors).toEqual([]);
  });

  it('splits EBAY_MCP_MEDIA_DIRS on the platform path delimiter and trims entries', () => {
    const access = getMediaAccessConfig({
      EBAY_MCP_MEDIA_DIRS: ` /srv/media ${path.delimiter} /var/uploads/ ${path.delimiter}`,
    });

    // Entries are normalized with path.resolve, which is drive-qualified on
    // Windows, so the expectations are derived the same way rather than pinned
    // to POSIX spellings.
    expect(access.allowedDirs).toEqual([path.resolve('/srv/media'), path.resolve('/var/uploads')]);
  });

  it('adds EBAY_MCP_MEDIA_ROOT to the allowed directories once', () => {
    const access = getMediaAccessConfig({
      EBAY_MCP_MEDIA_DIRS: '/srv/media',
      EBAY_MCP_MEDIA_ROOT: '/srv/media',
    });

    expect(access.mediaRoot).toBe(path.resolve('/srv/media'));
    expect(access.allowedDirs).toEqual([path.resolve('/srv/media')]);
  });

  it('rejects relative entries with an error naming the variable', () => {
    const access = getMediaAccessConfig({
      EBAY_MCP_MEDIA_DIRS: 'relative/dir',
      EBAY_MCP_MEDIA_ROOT: './media',
    });

    expect(access.allowedDirs).toEqual([]);
    expect(access.errors).toHaveLength(2);
    expect(access.errors[0]).toContain('EBAY_MCP_MEDIA_DIRS');
    expect(access.errors[1]).toContain('EBAY_MCP_MEDIA_ROOT');
  });
});

describe('ensureMediaDirs', () => {
  it('creates missing directories private to the user and leaves existing ones alone', async () => {
    const base = await mkdtemp(path.join(tmpdir(), 'media-ensure-'));
    const existing = path.join(base, 'existing');
    const missing = path.join(base, 'Claude', 'ebay-fotos');
    await mkdir(existing, { mode: 0o755 });

    const result = ensureMediaDirs({ allowedDirs: [existing, missing], errors: [] });

    expect(result).toEqual({ created: [missing], failed: [] });
    expect((await stat(missing)).isDirectory()).toBe(true);
    expect((await stat(missing)).mode & 0o777).toBe(0o700);
    expect((await stat(existing)).mode & 0o777).toBe(0o755);
  });

  it('does nothing when no media directory is configured', () => {
    expect(ensureMediaDirs({ allowedDirs: [], errors: [] })).toEqual({ created: [], failed: [] });
  });

  it('reports directories it cannot create instead of throwing', async () => {
    const base = await mkdtemp(path.join(tmpdir(), 'media-ensure-'));
    const blocker = path.join(base, 'file');
    await writeFile(blocker, 'x');

    const result = ensureMediaDirs({ allowedDirs: [path.join(blocker, 'sub')], errors: [] });

    expect(result.created).toEqual([]);
    expect(result.failed).toHaveLength(1);
  });
});
