import { Link } from 'react-router'

import { Card, EmptyState } from '../components/ui'

export default function NotFoundPage() {
  return (
    <div className="not-found-page">
      <Card>
        <EmptyState
          action={
            <Link className="button button--primary button--medium" to="/">
              Return to overview
            </Link>
          }
          description="The requested workspace route does not exist."
          title="Page not found"
        />
      </Card>
    </div>
  )
}
