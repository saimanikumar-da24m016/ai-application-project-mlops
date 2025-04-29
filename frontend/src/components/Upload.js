import { useState } from "react";
import { uploadImage } from "../api";

export default function Upload({ onResult }) {
  const [file, setFile] = useState(null);

  const handleSubmit = async () => {
    if (!file) return;
    const { data } = await uploadImage(file);
    onResult(data);
  };

  return (
    <div className="p-4 border-2 border-dashed rounded">
      <input
        type="file"
        accept="image/*"
        onChange={e => setFile(e.target.files[0])}
      />
      <button
        disabled={!file}
        onClick={handleSubmit}
        className="mt-2 px-4 py-2 bg-blue-600 text-white rounded"
      >
        Upload & Predict
      </button>
    </div>
  );
}

