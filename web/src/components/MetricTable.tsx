import React, { useState } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { Download, Search, Check, Copy } from "lucide-react";
import { MetricRowView } from "../types/start_schema";

interface MetricTableProps {
  rows: MetricRowView[];
  title?: string;
  onSelectEvidence?: (evidenceId: string) => void;
}

export const MetricTable: React.FC<MetricTableProps> = ({ rows, title, onSelectEvidence }) => {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");
  const [copiedCell, setCopiedCell] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCell(id);
    setTimeout(() => setCopiedCell(null), 1500);
  };

  const exportCSV = () => {
    const headers = ["Test ID", "Metric", "Value", "Unit", "Status", "Evidence ID"];
    const csvRows = [headers.join(",")];
    for (const r of rows) {
      csvRows.push(
        [
          `"${r.test_id}"`,
          `"${r.metric}"`,
          `"${r.value}"`,
          `"${r.unit || ""}"`,
          `"${r.status || "PASS"}"`,
          `"${r.evidence_id || ""}"`,
        ].join(",")
      );
    }
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `${title || "start_metrics"}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const columns: ColumnDef<MetricRowView>[] = [
    {
      accessorKey: "test_id",
      header: "Test Identifier",
      cell: (info) => <span className="font-mono text-xs text-foreground/90 font-medium">{String(info.getValue())}</span>,
    },
    {
      accessorKey: "metric",
      header: "Metric",
      cell: (info) => <span className="font-mono text-xs text-muted-foreground">{String(info.getValue())}</span>,
    },
    {
      accessorKey: "value",
      header: "Observed Value",
      cell: (info) => {
        const val = info.getValue();
        const formatted = typeof val === "number" ? (Number.isInteger(val) ? val.toString() : val.toFixed(4)) : String(val);
        const cellId = `${info.row.id}-val`;
        return (
          <div className="flex items-center gap-1.5 font-mono text-xs text-primary font-semibold">
            <span>{formatted}</span>
            <button
              onClick={() => copyToClipboard(formatted, cellId)}
              className="text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity"
              title="Copy value"
            >
              {copiedCell === cellId ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
            </button>
          </div>
        );
      },
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: (info) => {
        const st = String(info.getValue() || "PASS").toUpperCase();
        const isPass = st === "PASS" || st === "RECORDED";
        const isWarn = st === "WARN";
        return (
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold tracking-wider ${
              isPass
                ? "bg-green-950/60 text-green-400 border border-green-800/60"
                : isWarn
                ? "bg-amber-950/60 text-amber-400 border border-amber-800/60"
                : "bg-red-950/60 text-red-400 border border-red-800/60"
            }`}
          >
            {st}
          </span>
        );
      },
    },
    {
      accessorKey: "evidence_id",
      header: "Evidence Link",
      cell: (info) => {
        const evId = String(info.getValue() || "");
        if (!evId) return <span className="text-muted-foreground text-xs">—</span>;
        return (
          <button
            onClick={() => onSelectEvidence && onSelectEvidence(evId)}
            className="font-mono text-xs text-blue-400 hover:text-blue-300 hover:underline bg-blue-950/30 px-1.5 py-0.5 rounded border border-blue-900/50"
          >
            [{evId}]
          </button>
        );
      },
    },
  ];

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 8 } },
  });

  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden flex flex-col shadow-sm">
      {/* Header Controls */}
      <div className="p-3 border-b border-border bg-card/60 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {title && <h4 className="text-xs font-semibold text-foreground tracking-wide uppercase">{title}</h4>}
          <span className="text-[11px] text-muted-foreground font-mono">({rows.length} metrics)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Filter metrics..."
              value={globalFilter ?? ""}
              onChange={(e) => setGlobalFilter(e.target.value)}
              className="pl-8 pr-2.5 py-1 text-xs bg-background/80 border border-border rounded focus:outline-none focus:ring-1 focus:ring-primary w-40 text-foreground font-mono"
            />
          </div>
          <button
            onClick={exportCSV}
            className="flex items-center gap-1 px-2.5 py-1 text-xs font-mono bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded border border-border transition-colors"
          >
            <Download className="w-3 h-3" />
            <span>CSV</span>
          </button>
        </div>
      </div>

      {/* Table Body */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-border bg-muted/20">
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className="px-3.5 py-2 text-[11px] font-semibold text-muted-foreground uppercase font-mono tracking-wider cursor-pointer select-none hover:text-foreground"
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {{ asc: " ↑", desc: " ↓" }[header.column.getIsSorted() as string] ?? null}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-border/60">
            {table.getRowModel().rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-3.5 py-4 text-center text-xs text-muted-foreground font-mono">
                  No metrics found matching filter.
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="group hover:bg-muted/30 transition-colors">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-3.5 py-2 text-xs">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      {table.getPageCount() > 1 && (
        <div className="p-2 border-t border-border bg-card/40 flex items-center justify-between text-xs text-muted-foreground font-mono px-3">
          <span>
            Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
              className="px-2 py-0.5 rounded border border-border bg-secondary hover:bg-secondary/80 disabled:opacity-40"
            >
              Prev
            </button>
            <button
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
              className="px-2 py-0.5 rounded border border-border bg-secondary hover:bg-secondary/80 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
