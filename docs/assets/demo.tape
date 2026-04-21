# vhs demo tape for mente.
# Render with: `vhs docs/assets/demo.tape` (install: brew install vhs)
# Produces docs/assets/demo.gif (embed in README above the fold).

Output docs/assets/demo.gif

Set FontSize 14
Set Width 900
Set Height 540
Set Theme "Tokyo Night"
Set TypingSpeed 40ms
Set Padding 18

Hide
Type "./mente reset > /dev/null && ./mente" Enter
Sleep 600ms
Show

Sleep 1200ms

# Identity
Type "hello"
Enter
Sleep 700ms

# Memory capture
Type "remember that redis uses AOF for persistence"
Enter
Sleep 500ms

Type "remember that postgres uses WAL for durability"
Enter
Sleep 500ms

# Semantic search
Type "what do you know about databases?"
Enter
Sleep 1400ms

# Program synthesis
Type "compute the 15th fibonacci number"
Enter
Sleep 1800ms

# Library inspection
Type "/library"
Enter
Sleep 1600ms

# Self-model
Type "what have you been doing?"
Enter
Sleep 1800ms

Type "/quit"
Enter
Sleep 500ms
