package visitor.operation.miniJava.impl;

import java.lang.Integer;
import miniJava.visitor.miniJava.Block;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.Field;
import miniJava.visitor.miniJava.FieldBinding;
import miniJava.visitor.miniJava.Member;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.NewCall;
import miniJava.visitor.miniJava.NewObject;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.Parameter;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.SymbolBinding;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.NewObjectOperation;
import visitor.operation.miniJava.StateOperation;

public class NewObjectOperationImpl extends ExpressionOperationImpl implements NewObjectOperation {
	private final NewObject it;

	private final VisitorInterface vis;

	public NewObjectOperationImpl(NewObject it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		ObjectInstance res = ((ObjectInstance) (MiniJavaFactory.eINSTANCE.createObjectInstance()));
		res.setType(this.it.getType());
		state.getObjectsHeap().add(res);
		int i = ((int) (0));
		int z = ((int) (CollectionService.size(res.getType().getMembers())));
		while ((i) < (z)) {
			Member m = ((Member) (CollectionService.get(res.getType().getMembers(), i)));
			if(m instanceof Field) {
				Field f = ((Field) (m));
				if(!EqualService.equals((f.getDefaultValue()), (null))) {
					Value v = ((Value) (((ExpressionOperation)f.getDefaultValue().accept(vis)).evaluateExpression((State) (state))));
					FieldBinding fb = ((FieldBinding) (MiniJavaFactory.eINSTANCE.createFieldBinding()));
					fb.setField(f);
					fb.setValue(v);
					res.getFieldbindings().add(fb);
				}
			}
			i = ((Integer) ((i) + (1)));
		}
		i = ((Integer) (0));
		Method constructor = ((Method) (null));
		while ((((i) < (z)) && (EqualService.equals((constructor), (null))))) {
			Member m = ((Member) (CollectionService.get(res.getType().getMembers(), i)));
			if(m instanceof Method) {
				Method mtd = ((Method) (m));
				if(((EqualService.equals((mtd.getName()), (null))) && (EqualService.equals((CollectionService.size(mtd.getParams())), (CollectionService.size(this.it.getArgs())))))) {
					constructor = ((Method) (mtd));
				}
			}
			i = ((Integer) ((i) + (1)));
		}
		if(!EqualService.equals((constructor), (null))) {
			Context newContext = ((Context) (MiniJavaFactory.eINSTANCE.createContext()));
			i = ((Integer) (0));
			z = ((Integer) (CollectionService.size(this.it.getArgs())));
			while ((i) < (z)) {
				Expression arg = ((Expression) (CollectionService.get(this.it.getArgs(), i)));
				Parameter param = ((Parameter) (CollectionService.get(constructor.getParams(), i)));
				SymbolBinding binding = ((SymbolBinding) (MiniJavaFactory.eINSTANCE.createSymbolBinding()));
				binding.setSymbol(param);
				binding.setValue(((ExpressionOperation)arg.accept(vis)).evaluateExpression((State) (state)));
				i = ((Integer) ((i) + (1)));
				newContext.getBindings().add(binding);
			}
			NewCall call = ((NewCall) (MiniJavaFactory.eINSTANCE.createNewCall()));
			call.setNewz(this.it);
			((StateOperation)state.accept(vis)).pushNewFrame((ObjectInstance) (res), (NewCall) (call), (Context) (newContext));
			Block bd = ((Block) (constructor.getBody()));
			((BlockOperation)bd.accept(vis)).evaluateStatement((State) (state));
			((StateOperation)state.accept(vis)).popCurrentFrame();
		}
		ObjectRefValue tmp = ((ObjectRefValue) (MiniJavaFactory.eINSTANCE.createObjectRefValue()));
		tmp.setInstance(res);
		result = ((ObjectRefValue) (tmp));
		return result;
	}
}
