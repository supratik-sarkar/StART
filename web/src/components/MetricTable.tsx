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
      cell: (info) => (
        <span className="font-mono text-xs text-stone-900 font-medium">{String(info.getValue())}</span>
      ),
    },
    {
      accessorKey: "metric",
      header: "Metric",
      cell: (info) => (
        <span className="font-mono text-xs text-stone-600">{String(info.getValue())}</span>
      ),
    },
    {
      accessorKey: "value",
      header: "Observed Value",
      cell: (info) => {
        const val = info.getValue();
        const formatted =
          typeof val === "number"
            ? Number.isInteger(val)
              ? val.toString()
              : val.toFixed(4)
            : String(val);
        const cellId = `${info.row.id}-val`;
        return (
          <div className="flex items-center gap-1.5 font-mono text-xs text-indigo-600 font-semibold group">
            <span>{formatted}</span>
            <button
              onClick={() => copyToClipboard(formatted, cellId)}
              className="text-stone-400 hover:text-stone-700 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
              title="Copy value"
            >
              {copiedCell === cellId ? (
                <Check className="w-3 h-3 text-emerald-600" />
              ) : (
                <Copy className="w-3 h-3" />
              )}
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
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium tracking-wider ${
              isPass
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                : isWarn
                ? "bg-amber-50 text-amber-700 border border-amber-200"
                : "bg-rose-50 text-rose-700 border border-rose-200"
            }`}
          >
            {st}
          </span>
        );
      },
    },
    {
      accessorKey: "evidence_id",
      header: "Evidence ID",
      cell: (info) => {
        const evId = String(info.getValue() || "");
        if (!evId) return <span className="text-stone-300 font-mono text-xs">—</span>;
        return (
          <button
            onClick={() => onSelectEvidence && onSelectEvidence(evId)}
            className="font-mono text-xs text-indigo-600 hover:text-indigo-800 hover:underline font-medium cursor-pointer"
          >
            {evId}
          </button>
        );
      },
    },
  ];

  const table = useReactTable({
    data: rows,
    columns,
    state: {
      sorting,
      globalFilter,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: {
      pagination: {
        pageSize: 10,
      },
    },
  });

  return (
    <div className="flex flex-col h-full bg-white border border-[#E5E5E2] rounded-xl overflow-hidden shadow-xs text-left">
      {/* Table Action Bar */}
      <div className="p-3 border-b border-[#E5E5E2] bg-[#FBFBFA] flex items-center justify-between gap-4">
        <div className="flex items-center gap-2 flex-1 max-w-sm">
          <div className="relative w-full">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-stone-400" />
            <input
              type="text"
              value={globalFilter ?? ""}
              onChange={(e) => setGlobalFilter(e.target.value)}
              placeholder="Search metrics, tests, evidence..."
              className="w-full bg-white border border-[#E5E5E2] rounded-md pl-8 pr-3 py-1.5 text-xs text-stone-900 placeholder:text-stone-400 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-stone-500">{rows.length} records</span>
          <button
            onClick={exportCSV}
            className="px-2.5 py-1 bg-white hover:bg-stone-50 text-stone-700 text-xs font-medium border border-[#E5E5E2] rounded-md flex items-center gap-1.5 transition-colors cursor-pointer shadow-2xs"
          >
            <Download className="w-3 h-3 text-stone-500" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Table Container */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-left border-collapse">
          <thead className="bg-[#FBFBFA] sticky top-0 border-b border-[#E5E5E2] z-10">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className="px-4 py-2.5 text-[11px] font-semibold text-stone-600 font-mono select-none cursor-pointer hover:text-stone-900"
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-[#E5E5E2] bg-white text-stone-800">
            {table.getRowModel().rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="text-center py-8 text-stone-400 text-xs font-mono">
                  No matching metrics found.
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="hover:bg-stone-50/80 transition-colors">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-4 py-2.5">
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
      <div className="p-2.5 border-t border-[#E5E5E2] bg-[#FBFBFA] flex items-center justify-between text-xs font-mono text-stone-500">
        <div className="flex items-center gap-1">
          <span>Page</span>
          <span className="font-bold text-stone-800">{table.getState().pagination.pageIndex + 1}</span>
          <span>of</span>
          <span className="font-bold text-stone-800">{table.getPageCount() || 1}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            className="px-2 py-0.5 bg-white border border-[#E5E5E2] rounded text-stone-700 disabled:opacity-40 hover:bg-stone-50 cursor-pointer"
          >
            Prev
          </button>
          <button
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            className="px-2 py-0.5 bg-white border border-[#E5E5E2] rounded text-stone-700 disabled:opacity-40 hover:bg-stone-50 cursor-pointer"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};
