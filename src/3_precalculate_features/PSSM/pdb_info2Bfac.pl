#!/usr/bin/perl -w
#===============================================================================
#
#        Li Xue (), me.lixue@gmail.com
#        09/21/2016 10:25:11 PM
#
#  DESCRIPTION: extract information content from PSSM file and add it to pdb file.
#        INPUT (pssm file):
#           pdbresi pdbresn seqresi seqresn    A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V   IC
#                 1       M       1       L   -4   -4   -5   -6   -4   -3   -5   -6   -4   -1    1   -4   10   -3   -5   -4   -3   -4   -4   -2 2.32
#
#         Note: the following special case can happen (1acbE.pssm), where '-9-10' are connected
#          300 49 W   -8 -3 -9-10 -8 -7 -8 -8 -2 -8 -7 -8 -7 -2 -9 -8 -8 13 -2 -8
#
#       OUTPUT (pdb file with B-factor column):
#
#        USAGE: ./Info2Bfac.pl
#
#===============================================================================

use strict;
use warnings;
use warnings::register;
use utf8;
use File::Basename;

our %one2three = (
    'A'=>'ALA' ,
    'B'=>'ASX' ,
    'C'=>'CYS' ,
    'D'=>'ASP' ,
    'E'=>'GLU' ,
    'F'=>'PHE' ,
    'G'=>'GLY' ,
    'H'=>'HIS' ,
    'I'=>'ILE' ,
    'J'=>'XLE' ,
    'K'=>'LYS' ,
    'L'=>'LEU' ,
    'M'=>'MET' ,
    'N'=>'ASN' ,
    'O'=>'PYL' ,
    'P'=>'PRO' ,
    'Q'=>'GLN' ,
    'R'=>'ARG' ,
    'S'=>'SER' ,
    'T'=>'THR' ,
    'U'=>'SEC' ,
    'V'=>'VAL' ,
    'W'=>'TRP' ,
    'X'=>'XAA' ,
    'Y'=>'TYR' ,
    'Z'=>'GLX' ,
);


our %three2one = (
    'ALA' => 'A',
    'ASX' => 'B',
    'CYS' => 'C',
    'ASP' => 'D',
    'GLU' => 'E',
    'PHE' => 'F',
    'GLY' => 'G',
    'HIS' => 'H',
    'ILE' => 'I',
    'XLE' => 'J',
    'LYS' => 'K',
    'LEU' => 'L',
    'MET' => 'M',
    'ASN' => 'N',
    'PYL' => 'O',
    'PRO' => 'P',
    'GLN' => 'Q',
    'ARG' => 'R',
    'SER' => 'S',
    'THR' => 'T',
    'SEC' => 'U',
    'VAL' => 'V',
    'TRP' => 'W',
    'XAA' => 'X',
    'TYR' => 'Y',
    'GLX' => 'Z',
    'CYM' => 'C',
    'CSP' => 'C',
    'CYF' => 'C',
    'CFE' => 'C',
    'NEP' => 'H',
    'ALY' => 'K',
    'M3L' => 'K',
    'SEP' => 'S',
    'TOP' => 'T',
    'TYP' => 'Y',
    'PTR' => 'Y',
    'TYS' => 'Y',
    'HYP' => 'P',
    'PTR' => 'Y',
);

if (scalar @ARGV ne 3){
    print "\nUsage: perl Info2Bfac.pl pdbFL pssm_chnID_FL chnID \n\n";
    exit;
}

my $pdb_FL   = shift @ARGV;
my $pssmFL = shift @ARGV;
my $chnID = shift @ARGV;

my $basename =basename ($pdb_FL, '.pdb');
my $outputFL = "$basename.$chnID.Bfac.pdb";

#----- read pssms
my ( $pssm, $Info ) =
  &readPSSMfl_2($pssmFL, $chnID)
  ;    # $pssm->{PRO:A:197} = (2 -1 4 ... ) ; $Info->{PRO:A:197} =  0.02


#---- write output: pdb files with information content at the B-factor column

open( PDB, "<$pdb_FL" ) or die "Cannot open pdb file $pdb_FL: $!";
open( OUTPUT, ">$outputFL" )
  or die "Cannot open pdb file $outputFL: $!";

while (<PDB>) {
    s/[\r\n]//mg;
    if (/^ATOM\s+/) {

        my $aa         = substr( $_, 17, 3 );
        my $chnID_      = substr( $_, 21, 1 );
        my $atomResNum = substr( $_, 22, 4 );
        $atomResNum =~ s/\s+//g;
        my $key = "$aa:$chnID:$atomResNum";

        if ($chnID_ ne $chnID){
        # This chain is not what we are looking for
            next;
        }

        if ( $chnID_ eq $chnID) {

            if ( &isLigand($aa) ) {
                print "#warning: $aa is ligand\n";
                $Info->{$key} = 0;
            }

            if ( !defined $Info->{$key} ) {
                warnings::warn("Warning: $key has no information content in $pssmFL. -99 is assigned.");
                $Info->{$key} = -99;
            }

            my $n    = 6 - length( $Info->{$key} );
            my $Info_ = ' ' x $n;

            $Info_ = $Info_ . $Info->{$key};
            substr( $_, 60, 6, $Info_ );
        }

        print OUTPUT "$_\n";
    }
    else {
        print OUTPUT "$_\n";
    }
}

close PDB;
close OUTPUT;

print "$outputFL generated\n";



#------------------------------
sub readPSSMfl_2 {

# add chain ID to the key to pssm


    my $pssmFL = shift @_; #- 1EWY.protein1.ResNumPSSM
    my $chnID = shift @_; # A or B

#    print "Read pssm FL ($pssmFL) ... \n";

    my ($pssm,$Info) = &readPSSMfl_newformat($pssmFL);
    #-- $pssm->{PRO:197} = (2 -1 4 ... )
    #-- $Info->{PRO:197} = 0.23 #- information content

    #----- add chain ID to $pssm1_G1's keys: PRO:197 => PRO:A:197
    my $pssm_final;
    my $Info_final;
    foreach my $old_key (keys %$pssm){
        # $old_key = PRO:197
        my ($aa, $atomResNum) = split(/:/, $old_key);
        my $new_key = "$aa:$chnID:$atomResNum";

#        print "new_key: $new_key\n";

        $pssm_final->{$new_key}= $pssm->{$old_key};
        $Info_final->{$new_key}= $Info->{$old_key};
    }

    return ($pssm_final, $Info_final);

}

sub readPSSMfl_newformat {

#           pdbresi pdbresn seqresi seqresn    A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V   IC
#                 1       M       1       L   -4   -4   -5   -6   -4   -3   -5   -6   -4   -1    1   -4   10   -3   -5   -4   -3   -4   -4   -2 2.32

#Note: the following special case can happen (1acbE.pssm), where '-9-10' are connected
#  49 W   -8 -3 -9-10 -8 -7 -8 -8 -2 -8 -7 -8 -7 -2 -9 -8 -8 13 -2 -8

    my $pssmFL = shift @_;
    my %pssm;
    my $Info; #information content for each position

#    print "\n\nRead $pssmFL ... \n";
    open( INPUT, "<$pssmFL" ) || die("Cannot open $pssmFL:$!");

    while (<INPUT>) {
        s/[\n\r]//mg;

        if (/^\s*([\-\d]+\s+[a-zA-Z]{1}\s+\d+\s+[a-zA-Z]{1}\s+.+)/) {
        #  1       M       1       L   -4   -4   -5   -6   -4   -3   -5   -6   -4   -1    1   -4   10   -3   -5   -4   -3   -4   -4   -2 2.32


            my $line = $1;

            if ( $line =~ /\d+\-\d+/ ) {
                # '-9-10'
                # insert a space in front of -
                $line =~ s/-/ -/g;
            }

            my @a = split( /\s+/, $line );

            if (scalar @a ne 25){
                die("$pssmFL format is wrong! Check this line: \n$_:$!");
            }

            my $atomResNum = $a[0];
            my $aa =  $a[1]; # 'C'
            my $aa_3Lett = $one2three{$aa};

            if (!defined $aa_3Lett){
                die ("aa $aa is not a valid amino acid:$!");
            }

            @{ $pssm{"$aa_3Lett:$atomResNum"} } = @a[4 .. 23 ];

            my $Info_onePosition = pop @a;

            if (!defined $Info_onePosition){
                die ("Info not read for $aa_3Lett:$atomResNum:$!");
            }

            $Info -> {"$aa_3Lett:$atomResNum"} = $Info_onePosition;

        }

    }
    close INPUT;

    if ( !%pssm || !defined $Info ) {

        die("Nothing read from $pssmFL:$!")

    }

    return ( \%pssm, $Info);
}



sub readPSSMfl_oriformat {


# seqResNum, aa_1L, Last position-specific scoring matrix computed, weighted observed percentages rounded down, information per position, and relative weight of gapless real matches to pseudocounts
#            A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
#     1 D    -2  -2   1   7  -4  -1   2  -2  -2  -4  -4  -1  -4  -4  -2  -1  -2  -5  -4  -4    0   0   2  92   0   0   6   0   0   0   0   0   0   0   0   0   0   0   0   0  1.19 0.11
#     2 I     0  -2  -1  -2  -1  -1  -2  -2  -2   1  -1  -1  -1  -2  -2   1   5  -3  -2   0    3   0   0   0   0   0   0   0   0  13   0   0   0   0   0   7  73   0   0   3  0.51 0.10

#Note: the following special case can happen (1acbE.pssm), where '-9-10' are connected
#  49 W   -8 -3 -9-10 -8 -7 -8 -8 -2 -8 -7 -8 -7 -2 -9 -8 -8 13 -2 -8

    my $pssmFL = shift @_;
    my %pssm;
    my $Info; #information content for each position

#    print "\n\nRead $pssmFL ... \n";
    open( INPUT, "<$pssmFL" ) || die("Cannot open $pssmFL:$!");

    while (<INPUT>) {
        s/[\n\r]//mg;

        if (/^\s*(\d+\s+[a-zA-Z]{1}\s+.+)/) {
        # 498     1 D    -2  -2   1   7  -4  -1   2  -2  -2  -4  -4  -1  -4  -4  -2  -1  -2  -5  -4  -4    0   0   2  92   0   0   6   0   0   0   0   0   0   0   0   0   0   0   0   0  1.19 0.11


            my $line = $1;

            if ( $line =~ /\d+\-\d+/ ) {

                # '-9-10'
                # insert a space in front of -
                $line =~ s/-/ -/g;
            }

            my @a = split( /\s+/, $line );

            my $seqResNum = $a[0];
            my $atomResNum = $seqResNum;
            my $aa =  $a[1]; # 'C'
            my $aa_3Lett = $one2three{$aa};

            if (!defined $aa_3Lett){
                die ("aa $aa is not a valid amino acid:$!");
            }

            @{ $pssm{"$aa_3Lett:$atomResNum"} } = @a[2 .. 21 ];

            pop @a; # remove the last column: relative weight of gaphless real matches to pseudocounts
            my $Info_onePosition = pop @a;

            if (!defined $Info_onePosition){
                die ("Info not read for $aa_3Lett:$atomResNum:$!");
            }

            $Info -> {"$aa_3Lett:$atomResNum"} = $Info_onePosition;

        }

    }
    close INPUT;

    if ( !%pssm || !defined $Info ) {

        die("Nothing read from $pssmFL:$!")

    }

    return ( \%pssm, $Info);
}


sub isLigand{
    our %three2one;
    my $aa = uc(shift @_);
    my $ans =1;

    if (length($aa) != 3){
        die("aa $aa should be 3-letter code:$!");
    }
    if (defined $three2one{$aa}){
        $ans =0;
    }

    return $ans;

}



