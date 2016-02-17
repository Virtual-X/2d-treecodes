divert(-1)
define(`forloop',
       `pushdef(`$1', `$2')_forloop(`$1', `$2', `$3', `$4')popdef(`$1')')

define(`_forloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', incr($1))_forloop(`$1', `$2', `$3', `$4')')')

define(`forrloop',
       `pushdef(`$1', `$2')_forrloop(`$1', `$2', `$3', `$4')popdef(`$1')')

define(`_forrloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', decr($1))_forrloop(`$1', `$2', `$3', `$4')')')

define(`BINOMIAL', `esyscmd(python binomial.py $1 $2)')

USAGE LUNROLL
$1 iteration variable
$2 iteration start
$3 iteration end
$4 body

define(LUNROLL, `forloop($1, $2, $3,`$4')')
define(RLUNROLL, `forrloop($1, $2, $3, `$4')')
define(`TMP', $1_$2)

#usage
#SEQ(`ciao j...', j, 0, 1, 2, 3)
#result:
#ciao 0...  
#ciao 1...  
#ciao 2...
#ciao 3...

define(SEQ,  `define(`BODY', `$1') PEEL2(shift($*))' )

define(PEEL2, `pushdef(`$1', `ITERV') EATLOOP(shift($*)) popdef(`$1')')

define(EATLOOP, `ifelse($1,,, `pushdef(`ITERV', $1) BODY popdef(`ITERV') EATLOOP(shift($*))')')

define(`REDUCEL',`
ifelse(eval($# < 3), 1,, $2 $1 $3;)' `ifelse(eval($# <= 3), 1,`',`REDUCEL($1, shift(shift(shift($*))))')')

define(`ODDREMOVE', `ifelse(eval($# <= 2), 1, ifelse(eval($# > 0),1,$1), `$1,ODDREMOVE(shift(shift($*)))')')

define(`REDUCE',`REDUCEL($*)' `ifelse(eval($# <= 3), 1, ,`
REDUCE($1, ODDREMOVE(shift($*)))')')

#example: REDUCE(`+=', s0, s1, s2, s3, s4, s5, s6, s7)


divert(0)

