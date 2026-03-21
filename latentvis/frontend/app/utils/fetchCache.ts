"use client";

export function clearJsonFetchCache(prefix?: string) {
  void prefix;
}

export function fetchJsonCached(url: string): Promise<any> {
  return fetch(String(url), { cache: "no-store" }).then(async (res) => {
    if (!res.ok) {
      throw new Error(`fetch_json_failed:${res.status}:${url}`);
    }
    return res.json();
  });
}
