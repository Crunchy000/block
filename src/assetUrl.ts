/**
 * A file the page loads (from public/), with the build in its URL: browsers and GitHub Pages
 * cache these files by URL, so a new build's are fetched afresh rather than an old copy used.
 */
export const assetUrl = (path: string): string =>
  new URL(`${path}${path.includes('?') ? '&' : '?'}v=${__BUILD__}`, document.baseURI).toString();
