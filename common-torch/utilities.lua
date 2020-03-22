function findNode( gmodule, nodeName )
	for u,v in pairs( gmodule.forwardnodes ) do 
		local name = v.data.annotations.name
		if name and name == nodeName then 
			return v.data.module
		end
	end
end

function round(num, idp)
	local mult = 10^(idp or 0)
	return math.floor(num * mult + 0.5) / mult
end

function table.clone( t ) 

	local x = {} 

	for u,v in pairs( t ) do x[u] = v end 

	return x 

end

function table.insertTable( to, from ) 

	for i = 1, #from do 

		table.insert( to, from[i] ) 

	end

end
