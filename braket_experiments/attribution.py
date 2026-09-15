"""Amazon Braket attribution initializer.

Import this module before any Braket SDK import in every script that uses
the SDK. It tags AWS API calls as originating from Amazon Braket skill
usage, per the amazon-braket skill guidance.
"""

import botocore


def _braket_attribution(session):
    session.user_agent_extra = (
        f"{session.user_agent_extra} AWSSkill-Braket/1.0.0".strip()
    )


botocore.register_initializer(_braket_attribution)
