"""One-off CLI for provisioning accounts that must never exist via a public API route.

There is no public "sign up as admin" endpoint — that would be a security hole. This is
the only way an admin account gets created. It's also how you deliberately attach a first
recruiter login to one of the pre-existing "legacy" companies (seeded before auth existed)
for testing continuity — never automatic, never an auto-generated password.

Usage:
    python scripts/create_user.py --role admin --email admin@arclent.internal \
        --password "..." --name "Ops Admin"

    python scripts/create_user.py --role recruiter --company-id 1 \
        --email recruiter@zara.example --password "..." --name "Zara Recruiter"
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.auth import hash_password
from backend.database import create_admin_or_attached_user, get_company_profile_by_id, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["admin", "recruiter"], required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--company-id",
        type=int,
        default=None,
        help="Required for --role recruiter (attach to an existing company_profile row). Ignored for --role admin.",
    )
    args = parser.parse_args()

    if len(args.password) < 8:
        print("Password must be at least 8 characters.", file=sys.stderr)
        raise SystemExit(1)

    email = args.email.strip().lower()
    company_id = None
    if args.role == "recruiter":
        if args.company_id is None:
            print("--company-id is required for --role recruiter.", file=sys.stderr)
            raise SystemExit(1)
        profile = get_company_profile_by_id(args.company_id)
        if not profile:
            print(f"No company_profile row with id={args.company_id}.", file=sys.stderr)
            raise SystemExit(1)
        company_id = args.company_id

    init_db()
    try:
        user = create_admin_or_attached_user(
            email=email,
            password_hash=hash_password(args.password),
            name=args.name.strip(),
            company_id=company_id,
            role=args.role,
        )
    except sqlite3.IntegrityError:
        print(f"A user with email {email!r} already exists.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Created {user['role']} user id={user['id']} email={user['email']!r}"
          + (f" company_id={company_id}" if company_id else ""))


if __name__ == "__main__":
    main()
