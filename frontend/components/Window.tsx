"use client";
import React from "react";

interface WindowProps {
  id: string;
  title: string;
  onClose: (id: string) => void;
  children: React.ReactNode;
  totalWindows?: number;
}

export default function Window({ id, title, onClose, children, totalWindows = 1 }: WindowProps) {
  const width =
    totalWindows <= 1
      ? "100%"
      : "calc(50% - 0.5rem)";
  const height = totalWindows <= 2 ? "100vh" : "50vh";

  return (
    <div
      className="bg-white rounded shadow-md p-3 transition-all duration-300 ease-in-out box-border"
      style={{ width, height }}
    >
      <div className="flex justify-between items-center border-b pb-1 mb-2">
        <h3 className="font-semibold">{title}</h3>
        <button
          className="text-red-500 hover:text-red-700"
          onClick={() => onClose(id)}
        >
          ✕
        </button>
      </div>
      <div>{children}</div>
    </div>
  );
}