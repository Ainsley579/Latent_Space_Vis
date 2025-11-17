"use client";
import React, { useState } from "react";
import Slider from "@mui/material/Slider";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export default function ControlPanel({ onChange }) {
  const [dataSize, setDataSize] = useState(5000);
  const [featureSize, setFeatureSize] = useState(100);
  const [method, setMethod] = useState("umap");
  const [colorLegend, setColorLegend] = useState("cluster");

  const fireChange = (ds, fs, m, legend) => {
    onChange && onChange({ dataSize: ds, featureSize: fs, method: m, colorLegend: legend });
  };

  return (
    <Box
      sx={{
        p: 2,
        borderBottom: "1px solid #ddd",
        mb: 2,
        bgcolor: "#fafafa",
        borderRadius: 1,
        display: "flex",
        alignItems: "center",
        gap: 2,
      }}
    >
      <Box sx={{ minWidth: 180, mr: 2 }}>
        <Typography variant="body2">Data Size: {dataSize}</Typography>
        <Box sx={{ width: 120 }}>
          <Slider
            value={dataSize}
            min={0}
            max={10000}
            onChange={(_, v) => {
              setDataSize(v as number);
              fireChange(v as number, featureSize, method, colorLegend);
            }}
            size="small"
          />
        </Box>
      </Box>

      <Box sx={{ minWidth: 180, mr: 2 }}>
        <Typography variant="body2">Feature Count: {featureSize}</Typography>
        <Box sx={{ width: 120 }}>
          <Slider
            value={featureSize}
            min={0}
            max={500}
            onChange={(_, v) => {
              setFeatureSize(v as number);
              fireChange(dataSize, v as number, method, colorLegend);
            }}
            size="small"
          />
        </Box>
      </Box>
    </Box>
  );
}