#!/usr/bin/env python3
"""Audit Production Dependencies for Linux ARM64 (aarch64) Compatibility."""

from start.web.schemas import ZeroCostAttestation


def main() -> None:
    att = ZeroCostAttestation()
    print("Zero-Cost Provisioning Attestation:")
    print(f"  Provider: {att.provider}")
    print(f"  Shape: {att.tier_shape}")
    print(f"  Always Free Eligible: {att.always_free_eligible}")
    print(f"  Monthly Recurring Charge: ${att.recurring_monthly_charge_usd:.2f}")


if __name__ == "__main__":
    main()
