"use client";

import { useEffect, useState } from "react";
import EvidencePanel from "@/components/EvidencePanel";
import Modal from "@/components/Modal";
import { EventPage, RecEvent, api, fileUrl, fmtTime } from "@/lib/api";

const PAGE_SIZE = 20;

export default function HistoryPage() {
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<EventPage | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [why, setWhy] = useState<RecEvent | null>(null);

  // Any filter change goes back to page 1.
  useEffect(() => setPage(1), [q, status, from, to]);

  useEffect(() => {
    const params = new URLSearchParams({ page: String(page), page_size: String(PAGE_SIZE) });
    if (q.trim()) params.set("q", q.trim());
    if (status) params.set("status", status);
    // datetime-local has no timezone: interpret as local time, send as UTC ISO.
    if (from) params.set("from", new Date(from).toISOString());
    if (to) params.set("to", new Date(to).toISOString());

    let stale = false;
    const t = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await api<EventPage>(`/api/events?${params}`);
        if (!stale) {
          setData(res);
          setError("");
        }
      } catch (e) {
        if (!stale) setError((e as Error).message);
      } finally {
        if (!stale) setLoading(false);
      }
    }, 250);
    return () => {
      stale = true;
      clearTimeout(t);
    };
  }, [q, status, from, to, page]);

  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  return (
    <>
      <h1>Recognition history</h1>
      <div className="card">
        <div className="filters">
          <div>
            <label>Person</label>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Name or ID" />
          </div>
          <div>
            <label>Status</label>
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">All</option>
              <option value="known">Known</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>
          <div>
            <label>From</label>
            <input type="datetime-local" value={from} onChange={(e) => setFrom(e.target.value)} />
          </div>
          <div>
            <label>To</label>
            <input type="datetime-local" value={to} onChange={(e) => setTo(e.target.value)} />
          </div>
          <div>
            <button
              className="ghost"
              onClick={() => {
                setQ("");
                setStatus("");
                setFrom("");
                setTo("");
              }}
            >
              Clear filters
            </button>
          </div>
        </div>

        {error && <p className="error">{error}</p>}
        <table style={{ opacity: loading ? 0.6 : 1 }}>
          <thead>
            <tr>
              <th>Snapshot</th>
              <th>Time</th>
              <th>Person</th>
              <th>Status</th>
              <th>Confidence</th>
              <th>Camera</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data?.items.map((e) => (
              <tr key={e.id}>
                <td>
                  {e.snapshot_path ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img className="thumb" src={fileUrl(e.snapshot_path)} alt="" />
                  ) : (
                    <div className="thumb" />
                  )}
                </td>
                <td>{fmtTime(e.ts)}</td>
                <td>
                  {e.person_name ?? <span className="muted">Unknown</span>}
                  {e.person_code && <span className="muted"> {e.person_code}</span>}
                </td>
                <td>
                  <span className={`badge ${e.status}`}>{e.status}</span>
                </td>
                <td>{(e.confidence * 100).toFixed(1)}%</td>
                <td>{e.camera_id}</td>
                <td>
                  {e.has_evidence ? (
                    <button className="ghost" onClick={() => setWhy(e)}>
                      Why
                    </button>
                  ) : (
                    <span className="muted small">No data</span>
                  )}
                </td>
              </tr>
            ))}
            {data && data.items.length === 0 && (
              <tr>
                <td colSpan={7} className="muted">
                  No events found
                </td>
              </tr>
            )}
          </tbody>
        </table>

        <div className="pager">
          <span className="muted">{data?.total ?? 0} events</span>
          <button className="ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
            Prev
          </button>
          <span>
            Page {page} of {pages}
          </span>
          <button className="ghost" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
            Next
          </button>
        </div>
      </div>

      {why && (
        <Modal title={`Decision for ${why.person_name ?? "Unknown"}`} onClose={() => setWhy(null)}>
          <EvidencePanel eventId={why.id} />
        </Modal>
      )}
    </>
  );
}
