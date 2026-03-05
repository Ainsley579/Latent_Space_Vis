"use client";

const jsonPromiseCache = new Map<string, Promise<any>>();

export function clearJsonFetchCache(prefix?: string) {
  if (!prefix) {
    jsonPromiseCache.clear();
    return;
  }
  Array.from(jsonPromiseCache.keys()).forEach((k) => {
    if (k.startsWith(prefix)) jsonPromiseCache.delete(k);
  });
}

export function fetchJsonCached(url: string): Promise<any> {
  const key = String(url);
  const cached = jsonPromiseCache.get(key);
  if (cached) return cached;
  const req = fetch(key).then(async (res) => {
    if (!res.ok) {
      throw new Error(`fetch_json_failed:${res.status}:${key}`);
    }
    return res.json();
  });
  jsonPromiseCache.set(key, req);
  req.catch(() => {
    // Avoid caching failed responses forever.
    jsonPromiseCache.delete(key);
  });
  return req;
}

