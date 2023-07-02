import os


def is_mot_sequence(directory: str) -> bool:
    """Checks whether the given directory is a mot sequence.

    Returns
    -------
    bool
        Returns True if directory is a mot sequence and False otherwise.
    """

    directory_content = set(os.listdir(directory))

    return ('det' in directory_content
            or 'gt' in directory_content
            or 'seqinfo.txt' in directory_content) \
        and 'img1' in directory_content
