package visitor.operation.miniJava.impl;

import java.lang.String;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecoretools.ale.compiler.lib.LogService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ValueOperation;

public class ValueOperationImpl implements ValueOperation {
	private final Value it;

	private final VisitorInterface vis;

	public ValueOperationImpl(Value it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}

	public String customToString() {
		String result;
		result = ((String) ("ERROR customToString not overloaded"));
		return result;
	}

	public Value copyj() {
		Value result;
		LogService.log("ERROR copyj not implemented");
		result = ((Value) (null));
		return result;
	}
}
