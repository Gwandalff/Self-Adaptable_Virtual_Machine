package visitor.operation.miniJava.impl;

import java.lang.Integer;
import java.lang.String;
import miniJava.visitor.miniJava.ArrayRefValue;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ArrayRefValueOperation;
import visitor.operation.miniJava.ValueOperation;

public class ArrayRefValueOperationImpl extends ValueOperationImpl implements ArrayRefValueOperation {
	private final ArrayRefValue it;

	private final VisitorInterface vis;

	public ArrayRefValueOperationImpl(ArrayRefValue it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value copyj() {
		Value result;
		ArrayRefValue tmp = ((ArrayRefValue) (MiniJavaFactory.eINSTANCE.createArrayRefValue()));
		tmp.setInstance(this.it.getInstance());
		result = ((ArrayRefValue) (tmp));
		return result;
	}

	public String customToString() {
		String result;
		String res = ((String) ("["));
		int i = ((int) (0));
		int lgt = ((int) (CollectionService.size(this.it.getInstance().getValue())));
		while ((i) < (lgt)) {
			Value tmpv = ((Value) (CollectionService.get(this.it.getInstance().getValue(), i)));
			res = ((String) ((res) + (((ValueOperation)tmpv.accept(vis)).customToString())));
			if((i) < ((lgt) - (1))) {
				res = ((String) ((res) + (", ")));
			}
			i = ((Integer) ((i) + (1)));
		}
		res = ((String) ((res) + ("]")));
		result = ((String) (res));
		return result;
	}
}
