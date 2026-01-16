import { useEffect, useState } from 'react';
import type { FrameworkInfo } from '../services/api';
import { listFrameworks } from '../services/api';

interface FrameworkSelectorProps {
  selectedFramework: string;
  onFrameworkChange: (framework: string) => void;
}

export function FrameworkSelector({
  selectedFramework,
  onFrameworkChange,
}: FrameworkSelectorProps) {
  const [frameworks, setFrameworks] = useState<FrameworkInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadFrameworks() {
      try {
        const data = await listFrameworks();
        setFrameworks(data);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load frameworks');
        setLoading(false);
      }
    }

    loadFrameworks();
  }, []);

  if (loading) {
    return <div className="framework-selector loading">Loading frameworks...</div>;
  }

  if (error) {
    return <div className="framework-selector error">Error: {error}</div>;
  }

  const selected = frameworks.find((f) => f.name === selectedFramework);

  return (
    <div className="framework-selector">
      <label htmlFor="framework-select">
        <strong>Framework:</strong>
      </label>
      <select
        id="framework-select"
        value={selectedFramework}
        onChange={(e) => onFrameworkChange(e.target.value)}
      >
        {frameworks.map((framework) => (
          <option key={framework.name} value={framework.name}>
            {framework.displayName}
          </option>
        ))}
      </select>

      {selected && selected.capabilities && (
        <div className="framework-info">
          {(() => {
            const caps: string[] = [];
            if (selected.capabilities.supportsTools) caps.push('tools');
            if (selected.capabilities.supportsStreaming) caps.push('streaming');
            if (selected.capabilities.supportsMultiModal) caps.push('multi-modal');
            if (selected.capabilities.supportsMultiAgent) caps.push('multi-agent');
            if (selected.capabilities.supportsMemory) caps.push('memory');
            return caps.length > 0 ? caps.join(', ') : null;
          })()}
        </div>
      )}
    </div>
  );
}
