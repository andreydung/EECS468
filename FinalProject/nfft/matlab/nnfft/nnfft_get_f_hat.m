%NFFT_GET_F_HAT Get Fourier coefficients from plan
%   Copyright (c) 2002, 2015 Jens Keiner, Stefan Kunis, Daniel Potts

% Copyright (c) 2002, 2015 Jens Keiner, Stefan Kunis, Daniel Potts
%
% This program is free software; you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation; either version 2 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
% details.
%
% You should have received a copy of the GNU General Public License along with
% this program; if not, write to the Free Software Foundation, Inc., 51
% Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
%
% $Id: nfft_get_f_hat.m 3776 2012-06-03 13:29:25Z keiner $
function f_hat = nnfft_get_f_hat(p)

f_hat = nnfftmex('get_f_hat',p);
