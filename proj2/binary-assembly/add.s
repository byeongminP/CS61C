li s2 0 # Replace ??? with your register choice from part 1
li s1 0
jal Add
jal PrintRes

li s2 4 # Replace ??? with your register choice from part 1
li s1 0
jal Add
jal PrintRes

# ADD MORE TESTS HERE FOLLOWING THE STRUCTURE ABOVE

jal Exit

Add:
  # Place your Code here for +. Remember this is NOT a function call
  # You should perform all of your operations on the s1 and ??? register. You
  # should also assume that these are the only two values to operate on
  # (no nested expressions)


  add s1 s1 s2
  jr ra

PrintRes:
  mv a1 s1
  li a0 1
  ecall
  jr ra

Exit:
  li a0 10
  ecall
