import { useState } from "react";
import { sendFeedback } from "../api";

export default function Feedback({ imageId }) {
  const [label, setLabel] = useState("");
  const handleSubmit = async () => {
    await sendFeedback({ image_id: imageId, correct_label: label });
    alert("Thank you for your feedback!");
  };
  return (
    <div className="mt-4 space-y-2">
      <h3 className="font-medium">Is this wrong?</h3>
      <select
        value={label}
        onChange={e => setLabel(e.target.value)}
        className="border p-1"
      >
        <option value="">-- select correct label --</option>
        <option value="benign">Benign</option>
        <option value="Early Pre-B">Early Pre-B</option>
        <option value="Pre-B">Pre-B</option>
        <option value="Pro-B">Pro-B</option>
      </select>
      <button
        disabled={!label}
        onClick={handleSubmit}
        className="px-3 py-1 bg-green-600 text-white rounded"
      >
        Submit Feedback
      </button>
    </div>
  );
}

