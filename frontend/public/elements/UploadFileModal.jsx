import { useState } from "react";
import { Button } from "@/components/ui/button";

export default function UploadFileModal(props) {
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);

  const handleUpload = async () => {
    if (!file) return;

    // Convert file to base64 to send via payload
    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

    callAction({
      name: "confirm_upload_file",
      payload: { filename: file.name, file: base64 },
    });

    setUploaded(true);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxWidth: "400px" }}>
      <input
        type="file"
        onChange={(e) => !uploaded && setFile(e.target.files?.[0] || null)}
        disabled={uploaded} // make input uninteractive after upload
        style={{
          padding: "8px 10px",
          borderRadius: "6px",
          border: "1px solid #555",
          width: "100%",
          backgroundColor: uploaded ? "#2e2e2e" : "#1e1e1e",
          color: "#e5e5e5",
          cursor: uploaded ? "not-allowed" : "pointer",
        }}
      />

      {!uploaded && (
        <Button
          onClick={handleUpload}
          disabled={!file}
          variant="default"
          style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
        >
          Upload
        </Button>
      )}
    </div>
  );
}