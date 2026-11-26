import { colorSegmentTotal } from "../lib/objectSelection";

type ColorIndicatorChartProps = { segments: Record<string, number> };

export function ColorIndicatorChart({ segments }: ColorIndicatorChartProps) {
  const entries = Object.entries(segments);
  const total = colorSegmentTotal(segments);
  return (
    <div className="color-indicator" aria-label={`Indikator warna, total ${total}%`}>
      <div className="color-bars" aria-hidden="true">
        {entries.map(([name, value]) => (
          <span key={name} style={{ width: `${value}%` }} className={`color-${name}`} />
        ))}
      </div>
      <ul>
        {entries.map(([name, value]) => (
          <li key={name}>
            <span
              className={`color-${name}`}
              aria-hidden="true"
              style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%" }}
            />
            <span>{name}</span>
            <strong>{value}%</strong>
          </li>
        ))}
      </ul>
      <small>Indikator warna HSV, bukan diagnosis matang atau busuk.</small>
    </div>
  );
}
