import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export for deployment to static hosts (Vercel/Cloudflare/Netlify, etc.)
  output: "export",
  devIndicators: false,
};

export default nextConfig;
