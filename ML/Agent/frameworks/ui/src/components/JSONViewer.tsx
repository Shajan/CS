import { useState } from 'react';
import './JSONViewer.css';

interface JSONViewerProps {
  data: any;
  defaultExpanded?: boolean;
}

interface JSONNodeProps {
  keyName?: string;
  value: any;
  level: number;
  isLast: boolean;
  defaultExpanded?: boolean;
}

function JSONNode({ keyName, value, level, isLast, defaultExpanded = false }: JSONNodeProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded || level === 0);

  const indent = level * 20;

  if (value === null) {
    return (
      <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
        {keyName && <span className="json-key">"{keyName}"</span>}
        {keyName && <span className="json-colon">: </span>}
        <span className="json-null">null</span>
        {!isLast && <span className="json-comma">,</span>}
      </div>
    );
  }

  if (typeof value === 'boolean') {
    return (
      <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
        {keyName && <span className="json-key">"{keyName}"</span>}
        {keyName && <span className="json-colon">: </span>}
        <span className="json-boolean">{value.toString()}</span>
        {!isLast && <span className="json-comma">,</span>}
      </div>
    );
  }

  if (typeof value === 'number') {
    return (
      <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
        {keyName && <span className="json-key">"{keyName}"</span>}
        {keyName && <span className="json-colon">: </span>}
        <span className="json-number">{value}</span>
        {!isLast && <span className="json-comma">,</span>}
      </div>
    );
  }

  if (typeof value === 'string') {
    return (
      <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
        {keyName && <span className="json-key">"{keyName}"</span>}
        {keyName && <span className="json-colon">: </span>}
        <span className="json-string">"{value}"</span>
        {!isLast && <span className="json-comma">,</span>}
      </div>
    );
  }

  if (Array.isArray(value)) {
    const isEmpty = value.length === 0;

    if (isEmpty) {
      return (
        <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
          {keyName && <span className="json-key">"{keyName}"</span>}
          {keyName && <span className="json-colon">: </span>}
          <span className="json-bracket">[]</span>
          {!isLast && <span className="json-comma">,</span>}
        </div>
      );
    }

    return (
      <>
        <div
          className="json-line json-expandable"
          style={{ paddingLeft: `${indent}px` }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span className="json-arrow">{isExpanded ? '▼' : '▶'}</span>
          {keyName && <span className="json-key">"{keyName}"</span>}
          {keyName && <span className="json-colon">: </span>}
          <span className="json-bracket">[</span>
          {!isExpanded && <span className="json-preview"> {value.length} items </span>}
          {!isExpanded && <span className="json-bracket">]</span>}
          {!isExpanded && !isLast && <span className="json-comma">,</span>}
        </div>
        {isExpanded && (
          <>
            {value.map((item, index) => (
              <JSONNode
                key={index}
                value={item}
                level={level + 1}
                isLast={index === value.length - 1}
                defaultExpanded={defaultExpanded}
              />
            ))}
            <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
              <span className="json-bracket">]</span>
              {!isLast && <span className="json-comma">,</span>}
            </div>
          </>
        )}
      </>
    );
  }

  if (typeof value === 'object') {
    const keys = Object.keys(value);
    const isEmpty = keys.length === 0;

    if (isEmpty) {
      return (
        <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
          {keyName && <span className="json-key">"{keyName}"</span>}
          {keyName && <span className="json-colon">: </span>}
          <span className="json-brace">{'{}'}</span>
          {!isLast && <span className="json-comma">,</span>}
        </div>
      );
    }

    return (
      <>
        <div
          className="json-line json-expandable"
          style={{ paddingLeft: `${indent}px` }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span className="json-arrow">{isExpanded ? '▼' : '▶'}</span>
          {keyName && <span className="json-key">"{keyName}"</span>}
          {keyName && <span className="json-colon">: </span>}
          <span className="json-brace">{'{'}</span>
          {!isExpanded && <span className="json-preview"> {keys.length} keys </span>}
          {!isExpanded && <span className="json-brace">{'}'}</span>}
          {!isExpanded && !isLast && <span className="json-comma">,</span>}
        </div>
        {isExpanded && (
          <>
            {keys.map((key, index) => (
              <JSONNode
                key={key}
                keyName={key}
                value={value[key]}
                level={level + 1}
                isLast={index === keys.length - 1}
                defaultExpanded={defaultExpanded}
              />
            ))}
            <div className="json-line" style={{ paddingLeft: `${indent}px` }}>
              <span className="json-brace">{'}'}</span>
              {!isLast && <span className="json-comma">,</span>}
            </div>
          </>
        )}
      </>
    );
  }

  return null;
}

export default function JSONViewer({ data, defaultExpanded = false }: JSONViewerProps) {
  return (
    <div className="json-viewer">
      <JSONNode value={data} level={0} isLast={true} defaultExpanded={defaultExpanded} />
    </div>
  );
}
