import pretty_midi, os
midi = pretty_midi.PrettyMIDI(initial_tempo=120)
inst = pretty_midi.Instrument(program=0)
for i in range(20):
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60+i, start=i*0.25, end=i*0.25+0.2))
midi.instruments.append(inst)
midi.write("test_fixed.mid")
print("SIZE:", os.path.getsize("test_fixed.mid"))
print("NOTES:", len(inst.notes))
