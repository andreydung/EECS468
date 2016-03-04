/*
 * Copyright (c) 2002, 2015 Jens Keiner, Stefan Kunis, Daniel Potts
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* $Id: infft.h 3471 2010-04-08 21:43:41Z keiner $ */

#ifndef TICKS_H_
#define TICKS_H_

#include "cycle.h"

#define TICKS_PER_SECOND 2712186283.0

#ifndef HAVE_TICK_COUNTER
  typedef unsigned ticks;
  #define getticks() 0U;
  INLINE_ELAPSED(__inline__)
#endif

#endif /* TICKS_H_ */