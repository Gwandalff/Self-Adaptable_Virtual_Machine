package visitor.operation.miniJava.impl;

import java.lang.Boolean;
import miniJava.visitor.miniJava.Parameter;
import miniJava.visitor.miniJava.TypeRef;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ParameterOperation;
import visitor.operation.miniJava.TypeRefOperation;

public class ParameterOperationImpl extends SymbolOperationImpl implements ParameterOperation {
	private final Parameter it;

	private final VisitorInterface vis;

	public ParameterOperationImpl(Parameter it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public boolean compare(Parameter other) {
		boolean result;
		result = ((Boolean) (((EqualService.equals((this.it.getName()), (other.getName()))) && (((TypeRefOperation)this.it.getTypeRef().accept(vis)).compare((TypeRef) (other.getTypeRef()))))));
		return result;
	}
}
