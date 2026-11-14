-- Create the user service's own database on first init of a fresh
-- volume. The user service never shares the platform's
-- agentic_runtime database; it only shares the Postgres server.
-- Init scripts run once, so an existing volume needs
-- `createdb user_service` instead.
SELECT 'CREATE DATABASE user_service'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'user_service')\gexec
