/**
 * A file the page loads (from public/), with the build in its URL: browsers and GitHub Pages
 * cache these files by URL, so a new build's are fetched afresh rather than an old copy used.
 */
export const assetUrl = (path: string): string =>
  new URL(`${path}${path.includes('?') ? '&' : '?'}v=${__BUILD__}`, document.baseURI).toString();

/**
 * Whether a newer build is deployed than this page (version.json, written by each build, is
 * fetched past every cache): if so the page reloads into it, its URL carrying the new build
 * so the page itself isn't taken from the cache either. Resolves false if this is the newest
 * (or it can't tell, as in development).
 */
export async function reloadIfOutdated(): Promise<boolean> {
  try {
    const r = await fetch(new URL('version.json', document.baseURI), { cache: 'no-store' });
    if (!r.ok) return false;
    const { build } = (await r.json()) as { build?: string };
    const url = new URL(location.href);
    if (!build || build === __BUILD__ || url.searchParams.get('v') === build) return false; // (tried already: don't loop)
    url.searchParams.set('v', build);
    location.replace(url.toString());
    return true;
  } catch {
    return false;
  }
}
