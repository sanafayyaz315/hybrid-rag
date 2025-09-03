-- public.users definition
-- Drop table
-- DROP TABLE public.users;

CREATE TABLE public.users (
	id uuid NOT NULL,
	identifier text NOT NULL,
	metadata jsonb NOT NULL,
	"createdAt" text NULL,
	CONSTRAINT users_identifier_key UNIQUE (identifier),
	CONSTRAINT users_pkey PRIMARY KEY (id)
);

-- public.threads definition
-- Drop table
-- DROP TABLE public.threads;

CREATE TABLE public.threads (
	id uuid NOT NULL,
	"createdAt" text NULL,
	"name" text NULL,
	"userId" uuid NULL,
	"userIdentifier" text NULL,
	tags _text NULL,
	metadata jsonb NULL,
	CONSTRAINT threads_pkey PRIMARY KEY (id),
	CONSTRAINT "threads_userId_fkey" FOREIGN KEY ("userId") REFERENCES public.users(id) ON DELETE CASCADE
);

-- public.elements definition
-- Drop table
-- DROP TABLE public.elements;

CREATE TABLE public.elements (
	id uuid NOT NULL,
	"threadId" uuid NULL,
	"type" text NULL,
	url text NULL,
	"chainlitKey" text NULL,
	"name" text NOT NULL,
	display text NULL,
	"objectKey" text NULL,
	"size" text NULL,
	page int4 NULL,
	"language" text NULL,
	"forId" uuid NULL,
	mime text NULL,
	props jsonb NULL,
	CONSTRAINT elements_pkey PRIMARY KEY (id),
	CONSTRAINT "elements_threadId_fkey" FOREIGN KEY ("threadId") REFERENCES public.threads(id) ON DELETE CASCADE
);

-- public.feedbacks definition
-- Drop table
-- DROP TABLE public.feedbacks;

CREATE TABLE public.feedbacks (
	id uuid NOT NULL,
	"forId" uuid NOT NULL,
	"threadId" uuid NOT NULL,
	value int4 NOT NULL,
	"comment" text NULL,
	CONSTRAINT feedbacks_pkey PRIMARY KEY (id),
	CONSTRAINT "feedbacks_threadId_fkey" FOREIGN KEY ("threadId") REFERENCES public.threads(id) ON DELETE CASCADE
);

-- public.steps definition
-- Drop table
-- DROP TABLE public.steps;

CREATE TABLE public.steps (
	id uuid NOT NULL,
	"name" text NOT NULL,
	"type" text NOT NULL,
	"threadId" uuid NOT NULL,
	"parentId" uuid NULL,
	streaming bool NOT NULL,
	"waitForAnswer" bool NULL,
	"isError" bool NULL,
	metadata jsonb NULL,
	tags _text NULL,
	"input" text NULL,
	"output" text NULL,
	"createdAt" text NULL,
	"start" text NULL,
	"end" text NULL,
	generation jsonb NULL,
	"showInput" text NULL,
	"language" text NULL,
	"indent" int4 NULL,
	"defaultOpen" bool NOT NULL DEFAULT false,
	CONSTRAINT steps_pkey PRIMARY KEY (id),
	CONSTRAINT "steps_threadId_fkey" FOREIGN KEY ("threadId") REFERENCES public.threads(id) ON DELETE CASCADE
);
