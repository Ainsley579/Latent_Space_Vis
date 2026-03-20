import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  devIndicators: false,

  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;