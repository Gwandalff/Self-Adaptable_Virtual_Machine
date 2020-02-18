package visitor.operation.miniJava.impl;

import java.lang.Integer;
import miniJava.visitor.miniJava.Clazz;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.MethodCall;
import miniJava.visitor.miniJava.MethodCall2;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.Parameter;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.SymbolBinding;
import miniJava.visitor.miniJava.Value;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.MethodCallOperation;
import visitor.operation.miniJava.MethodOperation;
import visitor.operation.miniJava.StateOperation;

public class MethodCallOperationImpl extends ExpressionOperationImpl implements MethodCallOperation {
	private final MethodCall it;

	private final VisitorInterface vis;

	public MethodCallOperationImpl(MethodCall it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public Value evaluateExpression(State state) {
		Value result;
		ObjectRefValue realReceiver0 = ((ObjectRefValue) (((ObjectRefValue) (((ExpressionOperation)this.it.getReceiver().accept(vis)).evaluateExpression((State) (state))))));
		ObjectInstance realReceiver = ((ObjectInstance) (realReceiver0.getInstance()));
		Method realMethod = ((Method) (((Method) (((MethodOperation)this.it.getMethod().accept(vis)).findOverride((Clazz) (realReceiver.getType()))))));
		Context newContext = ((Context) (MiniJavaFactory.eINSTANCE.createContext()));
		int argsLength = ((int) (CollectionService.size(this.it.getArgs())));
		int i = ((int) (0));
		while ((i) < (argsLength)) {
			Expression arg = ((Expression) (CollectionService.get(this.it.getArgs(), i)));
			Parameter param = ((Parameter) (CollectionService.get(realMethod.getParams(), i)));
			SymbolBinding binding = ((SymbolBinding) (MiniJavaFactory.eINSTANCE.createSymbolBinding()));
			binding.setSymbol(param);
			binding.setValue(((ExpressionOperation)arg.accept(vis)).evaluateExpression((State) (state)));
			newContext.getBindings().add(binding);
			i = ((Integer) ((i) + (1)));
		}
		MethodCall2 call = ((MethodCall2) (MiniJavaFactory.eINSTANCE.createMethodCall2()));
		call.setMethodcall(this.it);
		((StateOperation)state.accept(vis)).pushNewFrame((ObjectInstance) (realReceiver), (MethodCall2) (call), (Context) (newContext));
		((MethodCallOperation)this.it.accept(vis)).call((Method) (realMethod), (State) (state));
		Value returnValue = ((Value) (((StateOperation)state.accept(vis)).findCurrentFrame().getReturnValue()));
		((StateOperation)state.accept(vis)).popCurrentFrame();
		result = ((Value) (returnValue));
		return result;
	}

	public void call(Method realMethod, State state) {
		((MethodOperation)realMethod.accept(vis)).call((State) (state));
	}
}
