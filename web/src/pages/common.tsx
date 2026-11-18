// Shared bits for pages: the "coming in 5.x" placeholder and a readable error state.
import { useState } from 'react';
import { Button, Card, EmptyState, SkeletonRows } from '../components';
import { useCreateProject } from '../api/hooks';
import { useCurrentProject } from '../api/project';
import { ApiError } from '../api/api';

export function ComingSoon({ task, what }: { task: string; what: string }) {
  return <EmptyState title={`Coming in ${task}`} body={what} className="py-16" />;
}

export function ErrorState({
  error,
  onRetry,
  notFound: notFoundTitle = 'This page isn’t here anymore.',
}: {
  error: unknown;
  onRetry?: () => void;
  /** What a 404 means on this screen ("This item isn’t here anymore."). */
  notFound?: string;
}) {
  const notFound = error instanceof ApiError && error.status === 404;
  const message =
    error instanceof Error && error.message ? error.message : 'Something went wrong loading this screen. Try again in a moment.';
  return (
    <div className="pt-2" role="alert">
      <Card className="text-center">
        <EmptyState
          className="py-6"
          title={notFound ? notFoundTitle : 'This didn’t load.'}
          body={notFound ? 'It may have been dismissed or moved. Go back and pick it again.' : message}
          action={
            onRetry && !notFound ? (
              <Button variant="secondary" onClick={onRetry}>
                Try again
              </Button>
            ) : undefined
          }
        />
      </Card>
    </div>
  );
}

/** For screens that need the current project: loading, failed, or none on this server yet. */
export function ProjectGate({ children, loadingLabel }: { children: React.ReactNode; loadingLabel: string }) {
  const { projectId, status, error, retry } = useCurrentProject();
  if (status === 'error') return <ErrorState error={error} onRetry={retry} />;
  if (status === 'empty') {
    return (
      <CreateProject />
    );
  }
  if (!projectId) return <SkeletonRows label={loadingLabel} count={4} />;
  return <>{children}</>;
}

export function Loading({ label }: { label: string }) {
  return (
    <div className="px-4 pt-4">
      <SkeletonRows label={label} count={4} />
    </div>
  );
}

/** First run on a fresh server: no project yet, so make one here instead of waiting for a seed. */
function CreateProject() {
  const create = useCreateProject();
  const { setProjectId } = useCurrentProject();
  const [name, setName] = useState('');
  const [address, setAddress] = useState('');
  const error = create.error instanceof Error ? create.error.message : null;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    const p = await create.mutateAsync({ name: name.trim(), address: address.trim() || null }).catch(() => null);
    if (p) setProjectId(p.id);
  }

  const inputCls =
    'h-12 w-full rounded-button border border-hairline bg-sheet px-3 text-body text-graphite outline-none focus:border-tape-ink';
  return (
    <form onSubmit={submit} className="flex flex-col gap-3 py-6">
      <h2 className="text-section font-medium text-graphite">Create your first project</h2>
      <p className="text-secondary text-pencil">A project holds your walks and punch list. Name it, then record or upload a walk.</p>
      <input className={inputCls} aria-label="Project name" placeholder="Project name (e.g. Unit 2B final punch)" value={name} onChange={(e) => setName(e.target.value)} />
      <input className={inputCls} aria-label="Address (optional)" placeholder="Address (optional)" value={address} onChange={(e) => setAddress(e.target.value)} />
      {error && <p role="alert" className="text-secondary text-graphite">{error}</p>}
      <Button type="submit" size="lg" block busy={create.isPending} busyLabel="Creating" disabled={!name.trim()}>
        Create project
      </Button>
    </form>
  );
}
