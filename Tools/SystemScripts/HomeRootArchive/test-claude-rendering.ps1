# Claude Code Text Rendering Test Script

Write-Host "`n=== Claude Code Text Rendering Test ===" -ForegroundColor Cyan

# Test 1: Basic Text
Write-Host "`n[Test 1] Basic Text Output:" -ForegroundColor Yellow
Write-Host "This is standard text output."

# Test 2: Colors
Write-Host "`n[Test 2] Color Support:" -ForegroundColor Yellow
Write-Host "Red Text" -ForegroundColor Red
Write-Host "Green Text" -ForegroundColor Green
Write-Host "Blue Text" -ForegroundColor Blue

# Test 3: Special Characters
Write-Host "`n[Test 3] Special Characters:" -ForegroundColor Yellow
Write-Host "Arrows: → ← ↑ ↓"
Write-Host "Symbols: ✓ ✗ ★ ♦ ♠ ♣ ♥"
Write-Host "Box Drawing: ┌─┐ │ └─┘"

# Test 4: Unicode
Write-Host "`n[Test 4] Unicode Support:" -ForegroundColor Yellow
Write-Host "Emoji: 🚀 💻 ✨ 🔧"
Write-Host "Math: ∑ ∏ √ ∞ ≈ ≠"
Write-Host "Greek: α β γ δ ε ζ"

# Test 5: Formatting
Write-Host "`n[Test 5] Text Formatting:" -ForegroundColor Yellow
Write-Host "Line 1 with`ttab`tseparated`ttext"
Write-Host "Line 2 with    spaces    between"

# Test 6: Multi-line
Write-Host "`n[Test 6] Multi-line Output:" -ForegroundColor Yellow
@"
This is a multi-line
text block that should
render properly with
line breaks preserved.
"@ | Write-Host

# Test 7: Progress Indicators
Write-Host "`n[Test 7] Progress Indicators:" -ForegroundColor Yellow
Write-Host "[##########] 100% Complete"
Write-Host "[#####     ] 50% In Progress"
Write-Host "[          ] 0% Not Started"

Write-Host "`n=== Test Complete ===" -ForegroundColor Green
Write-Host "If all text above renders correctly, your Claude Code text rendering is working properly." -ForegroundColor Cyan