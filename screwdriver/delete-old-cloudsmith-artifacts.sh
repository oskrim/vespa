#!/bin/bash
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

set -euo pipefail
MAX_NUMBER_OF_RELEASES=50

# Cloudsmith repo
rpm --import 'https://dl.cloudsmith.io/public/vespa/vespa/gpg.0F3DA3C70D35DA7B.key'
curl -1sLf 'https://dl.cloudsmith.io/public/vespa/vespa/config.rpm.txt?distro=el&codename=8' > /tmp/vespa-vespa.repo
dnf config-manager --add-repo '/tmp/vespa-vespa.repo'
rm -f /tmp/vespa-vespa.repo

VERSIONS_TO_DELETE=$(dnf list -y --quiet --showduplicates --disablerepo='*' --enablerepo=vespa-vespa vespa | awk '/[0-9].*\.[0-9].*\.[0-9].*/{print $2}' | sort -V | head -n -$MAX_NUMBER_OF_RELEASES | grep -v "7.594.36")

RPMS_TO_DELETE=$(mktemp)
trap "rm -f $RPMS_TO_DELETE" EXIT

for VERSION in $VERSIONS_TO_DELETE; do
  curl -sLf --header 'accept: application/json' \
    "https://api.cloudsmith.io/v1/packages/vespa/vespa/?query=version:${VERSION}" | jq -re '.[] | .slug' >> $RPMS_TO_DELETE
done

echo "Deleting the following RPMs:"
cat $RPMS_TO_DELETE

if [[ -n $SCREWDRIVER ]] && [[ -z $SD_PULL_REQUEST ]]; then
    for RPMID in $(cat $RPMS_TO_DELETE); do
      curl -sLf -u "$CLOUDSMITH_API_CREDS" -X DELETE \
        --header 'accept: application/json' \
        "https://api.cloudsmith.io/v1/packages/vespa/vespa/$RPMID/"
    done
fi

