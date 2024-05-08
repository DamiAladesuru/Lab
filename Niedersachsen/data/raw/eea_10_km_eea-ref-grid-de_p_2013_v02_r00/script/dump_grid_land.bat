@echo off
set ds=%~1
set gaul=%~2
set res=%~3

pgsql2shp -f %ds%_%res%km.shp -h iguana.eea.europa.eu -u michimau gis_sdi "select distinct g1.cellcode, g1.eoforigin, g1.noforigin, g1.the_geom from grids.inspire_grid_%res%km g1, (select distinct a.adm0_name, st_buffer(st_transform(a.wkb_geometry,3035),15000) as wkb_geometry from gaul.g2009_2008_0 a where a.adm0_name = '%gaul%') b where st_intersects(g1.the_geom, b.wkb_geometry)"
