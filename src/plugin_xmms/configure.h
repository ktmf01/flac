/* libxmms-flac - XMMS FLAC input plugin
 * Copyright (C) 2002  Daisuke Shimamura
 *
 * Based on mpg123 plugin
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#ifndef __CONFIGURE_H__
#define __CONFIGURE_H__

#include <glib.h>

typedef struct {
	gboolean tag_override;
	gchar *tag_format;
	gboolean convert_char_set;
	gchar *file_char_set;
	gchar *user_char_set;

	struct {
		gboolean enable;
		gint preamp;
		gboolean hard_limit;
		gboolean dither;
		guint noise_shaping; /* value must be one of NoiseShaping enum, c.f. plugin_common/replaygain_synthesis.h */
	} replaygain;
} flac_config_t;

extern flac_config_t flac_cfg;

extern void FLAC_XMMS__configure(void);
extern void FLAC_XMMS__aboutbox();

#endif



