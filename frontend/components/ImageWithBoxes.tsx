"use client";

import { PointerEvent, useRef, useState } from "react";

export type BoxItem = {
  id: number | string;
  box: number[]; // x1, y1, x2, y2 in the image's natural pixels
  kps?: number[][];
  label?: string;
  tone?: "default" | "selected" | "good" | "bad";
};

/**
 * An image with an SVG layer in the image's own pixel space. Shows boxes (+ optional
 * landmarks), and can let the user click a box or drag a new one. All coordinates are
 * natural-image pixels, so they can be sent to the backend as they are.
 */
export default function ImageWithBoxes({
  src,
  width,
  height,
  items,
  onSelect,
  onDraw,
  drawing,
  draft,
  frame,
}: {
  src: string;
  width: number;
  height: number;
  items: BoxItem[];
  onSelect?: (id: BoxItem["id"]) => void;
  onDraw?: (box: number[]) => void;
  drawing?: boolean; // true: dragging draws a new box
  draft?: number[] | null; // a user-drawn box to show
  frame?: boolean; // fixed square frame, image letterboxed (keeps panels the same size)
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [rect, setRect] = useState<number[] | null>(null);
  const start = useRef<[number, number] | null>(null);

  const toImage = (e: PointerEvent): [number, number] => {
    const r = svgRef.current!.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * width;
    const y = ((e.clientY - r.top) / r.height) * height;
    return [Math.min(Math.max(x, 0), width), Math.min(Math.max(y, 0), height)];
  };

  const down = (e: PointerEvent) => {
    if (!drawing || !onDraw) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    start.current = toImage(e);
    setRect(null);
  };
  const move = (e: PointerEvent) => {
    if (!start.current) return;
    const [x, y] = toImage(e);
    const [sx, sy] = start.current;
    setRect([Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]);
  };
  const up = () => {
    const r = rect;
    start.current = null;
    setRect(null);
    // ignore accidental clicks: need a real area (≥1.5% of the image's short side)
    if (r && r[2] - r[0] > Math.min(width, height) * 0.015 && r[3] - r[1] > Math.min(width, height) * 0.015)
      onDraw?.(r);
  };

  const fs = Math.max(width, height) / 38; // label size relative to the image
  const color = (t?: BoxItem["tone"]) =>
    t === "selected" ? "#4f8cff" : t === "good" ? "#3ecf6e" : t === "bad" ? "#ef5350" : "#f5c542";

  return (
    <div
      className={frame ? "iwb frame" : "iwb"}
      style={frame ? undefined : { aspectRatio: `${width} / ${height}` }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="" draggable={false} />
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio={frame ? "xMidYMid meet" : "none"}
        className={drawing ? "drawing" : ""}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
      >
        {items.map((it) => {
          const [x1, y1, x2, y2] = it.box;
          const c = color(it.tone);
          return (
            <g
              key={it.id}
              onClick={(e) => {
                if (onSelect) {
                  e.stopPropagation();
                  onSelect(it.id);
                }
              }}
              style={{ cursor: onSelect ? "pointer" : "default" }}
            >
              <rect
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill={it.tone === "selected" ? "rgba(79,140,255,.12)" : "transparent"}
                stroke={c}
                strokeWidth={2.5}
                vectorEffect="non-scaling-stroke"
                pointerEvents={drawing ? "none" : "all"}
              />
              {it.label && (
                <g pointerEvents="none">
                  <rect
                    x={x1}
                    y={Math.max(y1 - fs * 1.25, 0)}
                    width={fs * (0.62 * it.label.length + 0.9)}
                    height={fs * 1.25}
                    fill={c}
                  />
                  <text
                    x={x1 + fs * 0.35}
                    y={Math.max(y1 - fs * 1.25, 0) + fs * 0.95}
                    fontSize={fs}
                    fill="#fff"
                    fontWeight={600}
                  >
                    {it.label}
                  </text>
                </g>
              )}
              {it.kps?.map(([kx, ky], i) => (
                <circle
                  key={i}
                  cx={kx}
                  cy={ky}
                  r={Math.max(width, height) / 220}
                  fill="#00e5ff"
                  stroke="#000"
                  strokeWidth={1}
                  vectorEffect="non-scaling-stroke"
                  pointerEvents="none"
                />
              ))}
            </g>
          );
        })}
        {draft && (
          <rect
            x={draft[0]}
            y={draft[1]}
            width={draft[2] - draft[0]}
            height={draft[3] - draft[1]}
            fill="rgba(79,140,255,.15)"
            stroke="#4f8cff"
            strokeWidth={2.5}
            strokeDasharray="6 4"
            vectorEffect="non-scaling-stroke"
            pointerEvents="none"
          />
        )}
        {rect && (
          <rect
            x={rect[0]}
            y={rect[1]}
            width={rect[2] - rect[0]}
            height={rect[3] - rect[1]}
            fill="rgba(79,140,255,.2)"
            stroke="#fff"
            strokeWidth={2}
            strokeDasharray="5 4"
            vectorEffect="non-scaling-stroke"
            pointerEvents="none"
          />
        )}
      </svg>
    </div>
  );
}
