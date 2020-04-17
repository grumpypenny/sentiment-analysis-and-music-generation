import subprocess

def convert():
    command = "abcmidi/abc2midi.exe tempABC/song.abc -o outputMIDI/out.midi"

    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

    output, error = process.communicate()
    
    return output, error

if __name__ == "__main__":
    convert()
