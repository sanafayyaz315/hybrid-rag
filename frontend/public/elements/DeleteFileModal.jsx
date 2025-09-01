import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Trash } from "lucide-react";

export default function DeleteFileModal(props) {
  const [filename, setFilename] = useState(props.filename || "");
  const [deleted, setDeleted] = useState(false); // track if delete button was clicked

  const handleDelete = async () => {
    if (!filename) return;

    callAction({
      name: "confirm_delete_file",
      payload: { filename },
    });

    // After delete, mark as deleted to update UI
    setDeleted(true);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxWidth: "400px" }}>
      <input
        type="text"
        placeholder="Enter filename to delete"
        value={filename}
        onChange={(e) => !deleted && setFilename(e.target.value)} // editable only before deletion
        readOnly={deleted} // input becomes uninteractive after delete
        style={{
          padding: "8px 10px",
          borderRadius: "6px",
          border: "1px solid #555",
          width: "100%",
          backgroundColor: deleted
            ? "var(--background-color, #2e2e2e)" // indicate read-only state
            : "var(--background-color, #1e1e1e)",
          color: "var(--text-color, #e5e5e5)",
          cursor: deleted ? "not-allowed" : "text",
        }}
      />

      {!deleted && (
        <Button
          onClick={handleDelete}
          disabled={!filename}
          variant="default"
          style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
        >
          <Trash className="h-4 w-4" />
          Delete
        </Button>
      )}
    </div>
  );
}