export default function Result({ result }) {
    if (!result) return null;
    const { label, confidence, segmented_image_url } = result;
    return (
      <div className="mt-4 space-y-2">
        <h2 className="text-xl font-semibold">Prediction</h2>
        <p>
          <strong>{label}</strong> ({(confidence * 100).toFixed(1)}%)
        </p>
        <img
          src={segmented_image_url}
          alt="segmented"
          className="max-w-full border"
        />
      </div>
    );
  }
  