#!/usr/bin/env bash

# Bail out when commands fail.
set -e

# Move to the top of the SENSEI tree.
readonly output_base="$( pwd )"
cd "${BASH_SOURCE%/*}/../.."

info () {
    echo >&2 "$@"
}

die () {
    echo >&2 "$@"
    exit 1
}

usage () {
    die "$0: [(--tgz|--txz|--zip)...]" \
        "[--verbose] [-v <version>] [<tag>|<commit>]"
}

# Check for a tool to get SHA256 sums from.
if type -p sha256sum >/dev/null; then
    readonly sha256tool="sha256sum"
    readonly sha256regex="s/ .*//"
elif type -p cmake >/dev/null; then
    readonly sha256tool="cmake -E sha256sum"
    readonly sha256regex="s/ .*//"
else
    die "No 'sha256sum' tool found."
fi

compute_SHA256 () {
    local file="$1"
    readonly file
    shift

    $sha256tool "$file" | sed -e "$sha256regex"
}

# Check for a tool to get SHA512 sums from.
if type -p sha512sum >/dev/null; then
    readonly sha512tool="sha512sum"
    readonly sha512regex="s/ .*//"
elif type -p cmake >/dev/null; then
    readonly sha512tool="cmake -E sha512sum"
    readonly sha512regex="s/ .*//"
else
    die "No 'sha512sum' tool found."
fi

compute_SHA512 () {
    local file="$1"
    readonly file
    shift

    $sha512tool "$file" | sed -e "$sha512regex"
}

read_all_submodules () {
    # `git submodule foreach` clears GIT_INDEX_FILE from then environment
    # inside its command.
    local git_index="$GIT_INDEX_FILE"
    export git_index

    git submodule foreach --recursive --quiet '
        gitdir="$( git rev-parse --git-dir )"
        cd "$toplevel"
        GIT_INDEX_FILE="$git_index"
        export GIT_INDEX_FILE
        git add .gitmodules 2>/dev/null
        git rm --cached "$displaypath" >&2
        GIT_ALTERNATE_OBJECT_DIRECTORIES="$gitdir/objects" git read-tree -i --prefix="$displaypath/" "$sha1"
        echo "$gitdir/objects"
    ' | \
        tr '\n' ':'
}

read_submodules_into_index () {
    local object_dirs="$( git rev-parse --git-dir )/objects"
    local new_object_dirs

    while git ls-files -s | grep -q -e '^160000'; do
        new_object_dirs="$( read_all_submodules )"
        object_dirs="$object_dirs:$new_object_dirs"
    done

    object_dirs="$( echo "$object_dirs" | sed -e 's/:$//;s/^://' )"
    readonly object_dirs

    GIT_ALTERNATE_OBJECT_DIRECTORIES="$object_dirs"
    export GIT_ALTERNATE_OBJECT_DIRECTORIES
}

# Creates an archive of a git tree object.
git_archive () {
    local archive_format="$1"
    readonly archive_format
    shift

    local revision="$1"
    readonly revision
    shift

    local destination="$1"
    readonly destination
    shift

    local prefix
    if [ "$#" -gt 0 ]; then
        prefix="$1"
        shift
    else
        prefix="$destination"
    fi
    readonly prefix

    local ext
    local format
    local compress
    case "$archive_format" in
        tgz)
            ext=".tar.gz"
            format="tar"
            compress="gzip --best"
            ;;
        txz)
            ext=".tar.xz"
            format="tar"
            compress="xz --best"
            ;;
        zip)
            ext=".zip"
            format="zip"
            compress="cat"
            ;;
        *)
            die "unknown archive format: $format"
            ;;
    esac

    local output="$output_base/$destination$ext"
    readonly output

    local temppath="$output.tmp$$"
    readonly temppath

    git -c core.autocrlf=false archive $verbose "--format=$format" "--prefix=$prefix/" "$revision" | \
        $compress > "$temppath"
    mv "$temppath" "$output"


    info "Wrote $output"
    info "sha256: $(compute_SHA256 ${output})"
    info "sha512: $(compute_SHA512 ${output})"
}

#------------------------------------------------------------------------------

formats=
commit=
verbose=
version=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tgz)
            formats="$formats tgz"
            ;;
        --txz)
            formats="$formats txz"
            ;;
        --zip)
            formats="$formats zip"
            ;;
        --verbose)
            verbose="-v"
            ;;
        -v)
            shift
            version="$1"
            ;;
        --)
            shift
            break
            ;;
        -*)
            usage
            ;;
        *)
            if [ -z "$commit" ]; then
                commit="$1"
            else
                usage
            fi
            ;;
    esac
    shift
done

[ "$#" -eq 0 ] || \
    usage
[ -z "$commit" ] && \
    commit="HEAD"
[ -z "$formats" ] && \
    formats="tgz"

readonly formats
readonly verbose
readonly commit

git rev-parse --verify -q "$commit" >/dev/null || \
    die "'$commit' is not a valid commit"
if [ -z "$version" ]; then
    readonly desc="$( git describe "$commit" )"
    if ! [ "${desc:0:1}" = "v" ]; then
        die "'git describe $commit' is '$desc'; use -v <version>"
    fi
    version="${desc#v}"
    echo "$commit is version $version"
fi

readonly version

GIT_INDEX_FILE="$output_base/tmp-$$-index" && \
    trap "rm -f '$GIT_INDEX_FILE'" EXIT
export GIT_INDEX_FILE

result=0

info "Loading source tree from $commit..."
rm -f "$GIT_INDEX_FILE"
git read-tree -m -i "$commit"
git submodule sync --recursive
git submodule update --init --recursive
read_submodules_into_index
tree="$( git write-tree )"

info "Generating source archive(s)..."
for format in $formats; do
    git_archive "$format" "$tree" "SENSEI-$version" || \
        result=1
done

exit "$result"
