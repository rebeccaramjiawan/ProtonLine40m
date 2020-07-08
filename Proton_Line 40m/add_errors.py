use, sequence = TT40TT41;

!===============================================
!Static Errors
!===============================================


!Dipole errors

	!Vertical Bending
		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * bend_error;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = rbend;
		efcomp, order:=0,radius:=0.04,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	! Bending MBHC (bh1, bh2)
		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * mbhc_field;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = bh1;
		efcomp, order:=0,radius:=0.03,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * mbhc_field;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = bh2;
		efcomp, order:=0,radius:=0.03,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	! Bending MBG (mbs, b1, b1t)
		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * mbg_field;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = mbs;
		efcomp, order:=0,radius:=0.03,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * mbg_field;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = b1;
		efcomp, order:=0,radius:=0.03,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * mbg_field;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = b1t;
		efcomp, order:=0,radius:=0.03,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	! Bending B190 (b12_new, b2_new)
		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * b190_field;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = b12_new;
		efcomp, order:=0,radius:=0.03,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

		eoption, add = false, seed := ii+1;
		error_mb := tgauss(2) * b190_field;
		select, flag=ERROR, clear;
		select, flag=ERROR, class = b2_new;
		efcomp, order:=0,radius:=0.03,
		dknr:={error_mb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	!Bend Alignment Errors
		eoption, add=false, seed := ii+1;
		Select, flag=ERROR, clear;
		Select, flag=ERROR, class = rbend;
		ealign, dx:=tgauss(2) * bend_mis, dy:=gauss(2) * bend_mis, ds:=gauss(2) * bend_mis, dphi=bend_rot;


!Quad errors

	!Quad Field Errors
		eoption, add = true, seed := ii+1;
		error_qua := tgauss(2) * qtg_field;
		select, flag=error, clear;
		select, flag=error, class = quadrupole;
		efcomp, order:=1, radius:=0.04,
		dknr:={0,error_qua,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	!Quad Alignment Errors
		eoption, add=false, seed := ii+1;
		Select, flag=ERROR, clear;
		Select, flag=ERROR, class = quadrupole;
		ealign,
		dx:=tgauss(2) * quad_mis,
		dy:=tgauss(2) * quad_mis,
		ds:=tgauss(2) * quad_mis;


!BPM Errors
		eoption, add=false, seed := ii;
		Select, flag=ERROR, clear;
		Select, flag=ERROR, class=monitor;
		ealign,
		mrex:= bpm_error*tgauss(2);
		mrey:= bpm_error*tgauss(2);


!Error Output

	select, flag=error, clear;
    	select, flag=error, class = quadrupole;
    	select, flag=error, class= rbend;
	esave;

    select, flag = twiss, clear;
    !savebeta, label=start_line, place=electron_line$start, sequence=TT40TT41Seq;
        Select,flag=twiss,column=name,s,x,y;
        TWISS,
		centre,
		DELTAP=0.00,BETX= 27.931086,ALFX= 0.650549,DX= -0.557259,DPX= 0.013567,
		                 BETY=120.056512,ALFY=-2.705071,DY= 0.0,DPY= 0.0,
		                 x=0.00,px=0.00,y=0.0,py=0.0;
