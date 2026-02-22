from data_manager import clear_known_faces

if __name__ == '__main__':
    cleared = clear_known_faces()
    if cleared:
        print('All known faces cleared.')
    else:
        print('Failed to clear known faces.')
