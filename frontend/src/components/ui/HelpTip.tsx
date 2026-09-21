interface HelpTipProps {
  term: string
  children: string
}

export function HelpTip({ term, children }: HelpTipProps) {
  return (
    <span className="help-tip" tabIndex={0}>
      <span>{term}</span>
      <span className="help-tip__marker" aria-label={`Definition of ${term}`}>
        ?
      </span>
      <span className="help-tip__content" role="tooltip">
        {children}
      </span>
    </span>
  )
}
